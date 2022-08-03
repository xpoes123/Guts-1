import numpy as np
import nashpy as nash
from itertools import combinations
import joblib
#helper function for generalizedAlpha
def findPlayers(n,dropPlayers):
    players = []
    for i in range(n):
        if i not in dropPlayers:
            players = players + [i]
    return players
#helper function for generalizedAlpha
def helper(orderedStrats, n, weenie):
    value = 0
    if weenie:
        smallest = orderedStrats[0]
        for i in range(1, n):
            temp = np.repeat(1,n)
            temp[i] = 1 - n
            prob = smallest ** (n - 1) * (orderedStrats[i] - smallest)
            value = value + prob * temp
    indexList = np.array([i for i in range(n)])
    for dropNum in range(n):
        dropPlayerCombs = combinations(indexList, dropNum)
        holdNum = n - dropNum
        for dropPlayers in dropPlayerCombs:
            players = findPlayers(n, dropPlayers)
            maxStrat = orderedStrats[players[-1]]
            for loserNum in range(holdNum):
                fairNum = holdNum - loserNum
                loserPlayerCombs = combinations(players[:-1], loserNum)
                for losers in loserPlayerCombs:
                    temp = np.zeros(n)
                    prob = (1 - maxStrat) ** fairNum
                    for dropPlayer in dropPlayers:
                        prob *= orderedStrats[dropPlayer]
                        temp[dropPlayer] = -1 + holdNum - 1
                    for loser in losers:
                        prob *= (maxStrat - orderedStrats[loser])
                        temp[loser] = -1 - n + holdNum - 1
                    for player in players:
                        if player not in losers:
                            temp[player] = (-1 + n + holdNum - 1 + (fairNum - 1) * (-1 - n + holdNum - 1)) / fairNum
                    value = value + prob * temp
    return value

#weenie = True if weenie rule is activated
#strats is an array of pure strategies played by each player
#weenie = True if weenie rule is applied
#returns the payoff for player 1 (can easily be changed to return payoff for all players)
def alpha(strats, weenie = False):
    n = len(strats)

    orderedStrats = sorted(strats)
    value = helper(orderedStrats, n, weenie)
    sortedValue = np.zeros(n)
    for i in range(n):
        p = strats[i]
        pIndex = orderedStrats.index(p)
        sortedValue[i] = value[pIndex]
    return sortedValue[0]
#returns the expected future pot size
def beta(strats):
    n = len(strats)
    value = np.prod(strats)
    indexList = np.array([i for i in range(n)])
    for dropNum in range(n - 1):
        dropPlayerCombs = combinations(indexList, dropNum)
        holdnum = n - dropNum
        for dropPlayers in dropPlayerCombs:
            prob = 1
            players = findPlayers(n, dropPlayers)
            for dropPlayer in dropPlayers:
                prob *= strats[dropPlayer]
            for player in players:
                prob *= (1 - strats[player])
            value += (holdnum - 1) * prob
    return value

#get the next V_n for player 1
def getV(strats,Vlast):
  return alpha(strats)+beta(strats)*Vlast

#get the next V_n for the coalition
def getOppV(strats,Vlast):
  return -1*alpha(strats)+beta(strats)*Vlast

#get payoff matrix for player 1
def getA(n, N, M, Vprevious):
    A = np.zeros([N, M**(n - 1)])
    for i in range(N):
        for j in range(M ** (n - 1)):
            temp = [((j % (M ** (n - k))) // (M ** (n - 1 - k))) / (M - 1) for k in range(1, n)]
            value = getV(np.concatenate(([i/(N - 1)], temp)), Vprevious)
            A[i, j] = value
    return A

#get payoff matrix for the coalition
def getB(n, N, M, Vprevious):
    B = np.zeros([N, M ** (n - 1)])
    for i in range(N):
        for j in range(M ** (n - 1)):
            temp = [((j % (M ** (n - k))) // (M ** (n - 1 - k))) / (M - 1) for k in range(1, n)]
            value = getOppV(np.concatenate(([i / (N - 1)], temp)), Vprevious)
            B[i, j] = value
    return B

#get matrix beta
def getBeta(n, N, M):
    Beta = np.zeros([N, M ** (n - 1)])
    for i in range(N):
        for j in range(M ** (n - 1)):
            temp = [((j % (M ** (n - k))) // (M ** (n - 1 - k))) / (M - 1) for k in range(1, n)]
            value = beta(np.concatenate(([i / (N - 1)], temp)))
            Beta[i, j] = value
    return Beta

#returns the strategies played by "p2", the coalition against player 1
def getp2Strat(n, j, M):
    temp = [((j % (M ** (n - k))) // (M ** (n - 1 - k))) / (M - 1) for k in range(1, n)]
    return temp



#run fictitious play for n players for a certain number of iterations.
# N is the discretization for player 1; M is the discretization for each of players 2-n
def play(n, A, B, N=201, M=201, iterations=10000, Vprevious=0, VpreviousOpp=-2, saving=False, threshold=0.1):
    gts = nash.Game(A, B)

    np.random.seed(0)
    play_counts = tuple(gts.fictitious_play(iterations=iterations))
    row = play_counts[-1][0]
    col = play_counts[-1][1]
    row_halfway = play_counts[int(iterations / 2)][0]
    col_halfway = play_counts[int(iterations / 2)][1]
    p1_strats = []
    p1_usage = []
    opponent_strats = []
    opponent_usage = []
    p1_all_strats = []
    p1_all_usage = []
    opponent_all_strats = []
    opponent_all_usage = []
    if threshold >= 1:
        for i in range(N):
            p1_all_strats.append(i / (N - 1))
            p1_all_usage.append(row[i] - row_halfway[i])
            if row[i] - row_halfway[i] > threshold:
                p1_strats.append(i / (N - 1))
                p1_usage.append(row[i] - row_halfway[i])
        for i in range(M ** (n - 1)):
            opponent_all_strats.append(getp2Strat(n, i, M))
            opponent_all_usage.append(col[i] - col_halfway[i])
            if col[i] - col_halfway[i] > threshold:
                opponent_strats.append(getp2Strat(n, i, M))
                opponent_usage.append(col[i] - col_halfway[i])
    else:
        for i in range(N):
            if row[i] - row_halfway[i] > threshold:
                p1_strats.append(i / (N - 1))
                p1_usage.append(row[i] - row_halfway[i])
        for i in range(M ** (n - 1)):
            if col[i] - col_halfway[i] > threshold:
                opponent_strats.append(getp2Strat(n, i, M))
                opponent_usage.append(col[i] - col_halfway[i])
    if saving:
        joblib.dump(play_counts, "play_counts" + str(number) + ".txt")
    print("round number: " + str(number))
    print(p1_strats)
    print(p1_usage)
    print(opponent_strats)
    print(opponent_usage)
    Vsum = 0
    Asum = 0
    Bsum = 0
    if threshold >= 1:
        for i in range(len(p1_all_strats)):
            for j in range(len(opponent_all_strats)):
                Vsum += getV(np.concatenate(([p1_all_strats[i]], opponent_all_strats[j])), Vprevious) * \
                        p1_all_usage[i] * opponent_all_usage[j]
                Asum += alpha(np.concatenate(([p1_all_strats[i]], opponent_all_strats[j]))) * p1_all_usage[
                    i] * opponent_all_usage[j]
                Bsum += beta(np.concatenate(([p1_all_strats[i]], opponent_all_strats[j]))) * p1_all_usage[i] * \
                        opponent_all_usage[j]
    else:
        for i in range(len(p1_strats)):
            for j in range(len(opponent_strats)):
                Vsum += getV(np.concatenate(([p1_strats[i]], opponent_strats[j])), Vprevious) * p1_usage[i] * \
                        opponent_usage[j]
                Asum += alpha(np.concatenate(([p1_strats[i]], opponent_strats[j]))) * p1_usage[i] * \
                        opponent_usage[j]
                Bsum += beta(np.concatenate(([p1_strats[i]], opponent_strats[j]))) * p1_usage[i] * opponent_usage[
                    j]
    Asum = Asum / (int(iterations / 2) ** 2)
    Bsum = Bsum / (int(iterations / 2) ** 2)
    OppVsum = 0
    if threshold >= 1:
        for i in range(len(p1_all_strats)):
            for j in range(len(opponent_all_strats)):
                OppVsum += getOppV(np.concatenate(([p1_all_strats[i]], opponent_all_strats[j])),
                                   VpreviousOpp) * p1_all_usage[i] * opponent_all_usage[j]
    else:
        for i in range(len(p1_strats)):
            for j in range(len(opponent_strats)):
                OppVsum += getOppV(np.concatenate(([p1_strats[i]], opponent_strats[j])) , VpreviousOpp) * p1_usage[
                    i] * opponent_usage[j]

    Vsum = Vsum / (int(iterations / 2) ** 2)
    OppVsum = OppVsum / (int(iterations / 2) ** 2)
    print("average val: " + str(Vsum))
    print("average alpha: " + str(Asum))
    print("average beta: " + str(Bsum))
    print("Opponent average val: " + str(OppVsum))
    return (Vsum, OppVsum, "round number: " + str(number), p1_strats, p1_usage, opponent_strats, opponent_usage,
            "average val: " + str(Vsum), "Opponent average val: " + str(OppVsum), "average val: " + str(Vsum),
            "average alpha: " + str(Asum), "average beta: " + str(Bsum))


#edit filename as needed for how many players
filename = "4_player_results_asymmetric.txt"
open(filename, 'w').close()
N = 101
M = 101
n = 4
rounds = 100
curV = -1
oppV = -(n - 1)
initA = getA(n, N, M, 0)
initB = getB(n, N, M, 0)
A = np.copy(initA)
B = np.copy(initB)
Beta = getBeta(n, N, M)

# this simulates playing Guts for multiple rounds
for roundnum in range(rounds):
    A = initA + curV * Beta
    B = initB + oppV * Beta
    results = play(n, A, B, Vprevious=curV, VpreviousOpp=oppV, number=roundnum, iterations=1000, N = N, M= M)
    curV = results[0]
    oppV = results[1]
    file_object = open(filename, "a+")
    for i in range(2, len(results)):
        file_object.write(str(results[i]))
        file_object.write("\n")
    file_object.close()
    if curV <= -1:
        file_object = open(filename, "a+")
        print("player 1 leaves on round " + str(roundnum))
        file_object.write("player 1 leaves on round " + str(roundnum))
        file_object.close()
        break
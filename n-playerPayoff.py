import math
from itertools import combinations
import numpy as np

def findPlayers(n,dropPlayers):
    players = []
    for i in range(n):
        if i not in dropPlayers:
            players = players + [i]
    return players
def helper(orderedStrats, n):
    value = 0
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

#helper function for the weenie rule, returns the correction value
def p1Weenie(orderedStrats, index):
    n = len(orderedStrats)
    def getBins(num):
        return [(num % math.factorial(i + 1)) // math.factorial(i) for i in range(n)]
    def findWeenies(bins, highestBin):
        temp = 0
        for elem in bins:
            if elem == highestBin:
                temp +=1
        return temp
    binProbs = np.zeros(n)
    binProbs[0] = orderedStrats[0]
    for i in range(1, n):
        binProbs[i] = orderedStrats[i] - orderedStrats[i - 1]

    value = 0
    for i in range(0, math.factorial(n)):
        bins = getBins(i)
        prob = np.prod([binProbs[j] for j in bins])
        highestBin = np.amax(bins)
        if bins[index] == highestBin:
            fairNum = findWeenies(bins, highestBin)
            value += prob * ((fairNum - 1)/fairNum + (1 - n)/(fairNum))
        else:
            value += prob
    return value


def generalizedAlpha(strats, weenie = False):
    n = len(strats)
    orderedStrats = sorted(strats)
    value = helper(orderedStrats, n)
    sortedValue = np.zeros(n)
    for i in range(n):
        p = strats[i]
        pIndex = orderedStrats.index(p)
        sortedValue[i] = value[pIndex]
    if weenie:
        return sortedValue[0] + p1Weenie(orderedStrats, orderedStrats.index(strats[0]))
    return sortedValue[0]
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
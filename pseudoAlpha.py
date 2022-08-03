import math

import numpy as np

from itertools import combinations
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
def alpha(strats):
    n = len(strats)

    orderedStrats = sorted(strats)
    value = helper(orderedStrats, n)
    sortedValue = np.zeros(n)
    for i in range(n):
        p = strats[i]
        pIndex = orderedStrats.index(p)
        sortedValue[i] = value[pIndex]
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
def comb(n, k):
    return math.factorial(n)/(math.factorial(k) * math.factorial(n - k))

#payoff function for pseudo-bloc case
def pseudoAlpha(n, p1, p2, p3):
    value = 0

    #case 1: p1 < p2 < p3
    if p1 <= p2 and p2 <= p3:
        for dropNum in range(n - 1):
            holdnum = n - 2 - dropNum
            p3prob = p3 ** dropNum * (1 - p3) ** holdnum * comb(n - 2, dropNum)
            if dropNum == n - 2:
                #case a: h h
                value += (1 - p2) * (1 - p2) * p3prob * -(dropNum * (-1 + holdnum + 1))/(holdnum + 2)
                value += (p2 - p1) * (1 - p2) * p3prob * (-1 - n + holdnum + 1)
                #case b: h d
                value += (1 - p1) * p2 * p3prob * (-1 + n + holdnum)
                #case c: d h
                value += p1 * (1 - p2) * p3prob * (-1 + holdnum)
            else:
                #case a: h h
                value += (1 - p3) * (1 - p3) * p3prob * -(dropNum * (-1 + holdnum + 1))/(holdnum + 2)
                value += (p3 - p1) * (1 - p2) * p3prob * (-1 - n + holdnum + 1)
                value += (1 - p3) * (p3- p2) * p3prob * -(dropNum * (-1 + holdnum + 1) - n + holdnum)/(holdnum + 1)
                #case b: d h
                value += p1 * (1 - p2) * p3prob * (-1 + holdnum )
                #case c: h d
                value += (1 - p3) * p2 * p3prob * -((1 + dropNum) * (holdnum - 1))/(holdnum + 1)
                value += (p3 - p1) * p2 * p3prob * (-1 - n + holdnum)
                #case d: d d
                value += p1 * p2 * p3prob * (-1 + max(holdnum - 1, 0))
    #case 2: p1 < p3 < p2
    elif p1 <= p3 and p3 <= p2:
        for dropNum in range(n - 1):
            holdnum = n - 2 - dropNum
            p3prob = p3 ** dropNum * (1 - p3) ** holdnum * comb(n - 2, dropNum)
            if dropNum == n - 2:
                #case a: h h
                value += (p2 - p1) * p3prob * (1 - p2) * (-n)
                #case b: h d
                value += (1 - p1) * p2 * p3prob * (-1 + n)
                # case c: d h
                value += p1 * p3prob * (1 - p2) * (-1 + holdnum)
            else:
                for p3losers in range(holdnum + 1):
                    # case a: h h
                    p3fair = holdnum - p3losers
                    p3probFair = p3 ** dropNum * (1 - p2) ** p3fair * (p2 - p3)** p3losers * comb(holdnum, p3losers) * comb(n-2, dropNum)
                    value += (1 - p2) * p3probFair * (1 - p2) * -(dropNum * holdnum + p3losers*(holdnum - n))/(p3fair + 2)
                value += (p2 - p1) * p3prob * (1 - p2) * (-1 - n + holdnum + 1)
                #case b: h d
                value += (1 - p3) * p3prob * p2 * -((dropNum + 1) * (holdnum - 1))/(holdnum + 1)
                value += (p3 - p1) * p3prob * p2 * (-1 - n + holdnum)
                #case c: d h
                value += p1 * p3prob * (1 - p2) * (-1 + holdnum)
                #case d: d d
                value += p1 * p2 * p3prob * (-1 + max(holdnum - 1, 0))
    #case 3: p2 < p1 < p3:
    elif p2 <= p1 and p1 <= p3:
        for dropNum in range(n - 1):
            holdnum = n - 2 - dropNum
            p3prob = p3 ** dropNum * (1 - p3) ** holdnum * comb(n - 2, dropNum)
            if dropNum == n - 2:
                #case a: h h
                value += (1 - p1) * (1 - p1) * p3prob * -(dropNum * (-1 + holdnum + 1))/(holdnum + 2)
                value += (p1 - p2) * (1 - p1) * p3prob * (-1 + n + holdnum + 1)
                #case b: h d
                value += (1 - p2) *p1 * p3prob * (-1 + holdnum)
                #case c: d h
                value += p2 * (1 - p1) * p3prob * (-1 + n + holdnum)
            else:
                #case a: h h
                value += (1 - p3) * (1 - p3) * p3prob * -(dropNum * (-1 + holdnum + 1))/(holdnum + 2)
                value += (p3 - p1) * (1 - p2) * p3prob * (-1 - n + holdnum + 1)
                value += (1 - p3) * (p3- p2) * p3prob * -(dropNum * (-1 + holdnum + 1) - n + holdnum)/(holdnum + 1)
                #case b: d h
                value += p1 * (1 - p2) * p3prob * (-1 + holdnum)
                #case c: h d
                value += (1 - p3) * p2 * p3prob * -((1 + dropNum) * (holdnum - 1))/(holdnum + 1)
                value += (p3 - p1) * p2 * p3prob * (-1 - n + holdnum)
                #case d: d d
                value += p1 * p2 * p3prob * (-1 + max(holdnum - 1, 0))
    #case 4: p3 < p1 < p2:
    elif p3 <= p1 and p1 <= p2:
        for dropNum in range(n - 1):
            holdnum = n - 2 - dropNum
            p3prob = p3 ** dropNum * (1 - p3) ** holdnum * comb(n - 2, dropNum)
            if dropNum == n - 2:
                #case a: h h
                value += p3prob * (p2 - p1) * (1 - p2) * (-n)
                #case b: h d
                value += p3prob * (1 - p1) * p2 * (-1 + n)
                #case c: d h
                value += p3prob * p1 * (1 - p2) * (-1)
            else:
                for p3losers in range(holdnum + 1):
                    p3fair = holdnum - p3losers
                    p3probFair = p3 ** dropNum * (1 - p2) ** p3fair * (p2 - p3) ** p3losers * comb(holdnum, p3losers) * comb(n - 2, dropNum)
                    #case a: h h
                    value += p3probFair * (1 - p2) * (1 - p2) * -(dropNum * holdnum + p3losers*(holdnum - n))/(p3fair + 2)
                    # case b: h d
                    p3probFair = p3 ** dropNum * (1 - p1) ** p3fair * (p1 - p3) ** p3losers * comb(holdnum, p3losers) * comb(n - 2, dropNum)
                    value += p3probFair * (1 - p1) * p2 * -((dropNum + 1) * (-1 + holdnum) + p3losers * (holdnum - n - 1)) / (p3fair + 1)
                value += p3prob * (p2 - p1) * (1 - p2) * (holdnum - n)
                #case c: d h
                value += p3prob * p1 * (1 - p2) * (-1 + holdnum)
                #case d: d d
                value += p3prob * p1 * p2 * (-1 + max(holdnum - 1, 0))
    #case 5: p2 < p3 < p1:
    elif p2 <= p3 and p3 <= p1:
        for dropNum in range(n - 1):
            holdnum = n - 2 - dropNum
            p3prob = p3 ** dropNum * (1 - p3) ** holdnum * comb(n - 2, dropNum)
            if dropNum == n - 2:
                # case a: h h
                value += (p1 - p2) * p3prob * (1 - p1) * n
                # case b: h d
                value += (1 - p2) * p1 * p3prob * (-1)
                # case c: d h
                value += p2 * p3prob * (1 - p1) * (-1 + n)
            else:
                for p3losers in range(holdnum + 1):
                    # case a: h h
                    p3fair = holdnum - p3losers
                    p3probFair = p3 ** dropNum * (1 - p1) ** p3fair * (p1 - p3) ** p3losers * comb(holdnum, p3losers) * comb(n - 2, dropNum)
                    value += (1 - p1) * p3probFair * (1 - p1) * -(dropNum * holdnum + p3losers * (holdnum - n)) / (p3fair + 2)
                    value += (p1 - p2) * p3probFair * (1 - p1) * -(dropNum * holdnum + (p3losers + 1) * (holdnum - n)) / (p3fair + 1)
                    # case c: d h
                    value += p2 * p3probFair * (1 - p1) * -((dropNum + 1) * (holdnum - 1) + p3losers * (holdnum - n - 1)) / (p3fair + 1)
                # case b: h d
                value += (1 - p2) * p3prob * p1 * (-1 + holdnum)
                # case d: d d
                value += p1 * p2 * p3prob * (-1 + max(holdnum - 1, 0))
    #case 6: p3 < p2 < p1:
    else:
        for dropNum in range(n - 1):
            holdnum = n - 2 - dropNum
            p3prob = p3 ** dropNum * (1 - p3) ** holdnum * comb(n - 2, dropNum)
            if dropNum == n - 2:
                #case a: h h
                value += p3prob * (p1 - p2) * (1 - p1) * (n)
                #case b: h d
                value += p3prob * (1 - p2) * p1 * (-1)
                #case c: d h
                value += p3prob * p2 * (1 - p1) * (-1 + n)
            else:
                for p3losers in range(holdnum + 1):
                    p3fair = holdnum - p3losers
                    p3probFair = p3 ** dropNum * (1 - p1) ** p3fair * (p1 - p3) ** p3losers * comb(holdnum, p3losers) * comb(n - 2, dropNum)
                    #case a: h h
                    value += p3probFair * (1 - p1) * (1 - p1) * -(dropNum * holdnum + p3losers*(holdnum - n))/(p3fair + 2)
                    value += p3probFair * (p1 - p2) * (1 - p1) *-(dropNum * holdnum + (p3losers + 1)*(holdnum - n))/(p3fair + 1)
                    # case c: d h
                    value += p3probFair * p2 * (1 - p1) * -((dropNum + 1)* (holdnum - 1) + p3losers * (holdnum - n - 1))/(p3fair + 1)
                #case b: h d
                value += p3prob * (1 - p2) * p1 * (-1 + holdnum)
                #case d: d d
                value += p3prob * p1 * p2 * (-1 + max(holdnum - 1, 0))
    return value


def pseudoBeta(n, p1, p2, p3):
    value = p1 * p2 * p3 ** (n - 2)
    for dropNum in range(n - 1):
        holdNum = n - 2 - dropNum
        p3prob = (1 - p3) ** holdNum * p3 ** dropNum * comb(n - 2, dropNum)
        #case a: h h
        value += (1 - p1) * (1 - p2) * p3prob * (2 + holdNum - 1)
        #case b: h d
        value += (1 - p1) * p2 * p3prob * (1 + holdNum - 1)
        #case c: d h
        value += p1 * (1 - p2) * p3prob * (1 + holdNum - 1)
        #case d: d d
        value += p1 * p2 * p3prob * max(0, holdNum - 1)
    return value



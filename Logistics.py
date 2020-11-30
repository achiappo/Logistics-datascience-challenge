
__author__ = 'Andrea Chiappo'
__email__ = 'chiappo.andrea@gmail.com'

import numpy as np
from random import shuffle
from collections import Counter
from itertools import permutations

###############################################################################

def asset_demand(nassets, ntasks, nvalues, teamsize, probs=None):
    """
      Function to generate the daily assets' demands 
      For each asset, randomly generate the probability of good weather and 
      a number of tasks, labelling each task with two random values, 
      one indicating its importance and the other the team size required
    
      input:
       - nassets: number of assets whose demand(s) to generate
       - ntasks: number of possible task
       - nvalues: number of possible task values
       - teamsize: requested teamsize for each task
       - probs: probability of occurance of each task
    
      output:
       a list of dictionaries, each characterising a demanded task
    """
    if probs:
        assert len(probs)==nvalues, "length of probs must be equal to nvalues"
    
    day_demand = []
    for a in range(nassets):
        weather = np.random.rand()
        day_tasks_num = np.random.randint(ntasks)
        for k in range(day_tasks_num):
            val = np.random.choice(nvalues, p=probs)+1
            per = np.random.choice(teamsize)+1
            Kdict = dict(a=a+1, k=k+1, v=val, s=per, w=weather)
            day_demand.append(Kdict)
    
    return day_demand

###############################################################################

def staff_available(nvalues, mean=60, sigma=20, value_split=True):
    """
     Function to generate the daily staff available
    
     input:
      - nvalues : number of possible task values
      - mean : mean value of daily available staff distribution
      - sigma : spread of daily available staff distribution
      - split_value : option to group simulated available staff by value
    
     output:
      if split_value == True
        randomly split the available staff into groups
        each corresponding to only one task value
        return a list of dictionaries characterising each group
    
      if split_value == False
        return the whole number of available staff (no grouping)
    """

    valid = False
    while not valid:
        # The mean and standard deviation given can occasionally produce 
        # negative staff numbers, when drawing from a normal distribution
        # The present condition prevents this
        staff = int(np.random.normal(loc=mean, scale=sigma))
        if staff>0:
            valid = True
    
    if value_split:
        shares = np.random.rand(nvalues-1)        # randomly select nvalues shares
        shares /= sum(shares)                     # normalise to unity
        indexs = np.round(shares*100).astype(int)
        indexs.sort()

        day_staff = []
        groups = np.split(np.arange(staff)+1, indexs)
        # assign each share to a distinct role (task value)
        vals = np.random.choice(nvalues, nvalues, replace=False)+1
        for s,sg in enumerate(groups):
            if sg.size>0:
                per = len(sg)
                Sdict = dict(v=vals[s], s=per)
                day_staff.append(Sdict)

        # return list ordered in growing task value
        return sorted(day_staff, key=lambda d:d['v'])
    else:
        return staff

###############################################################################

def demand_staff_assign(demands, staff, nvalues, old=None, group_priority=True, value_priority=True):
    """
     Function meant to match daily assets' demand(s) with staff availability
     
     input=
      - demands        = list of dictionaries characterising each demand
      - staff          = available staff (either a whole number or a list of 
                                          dictionaries for every task)
      - nvalues        = number of possible task values
      - old            = possible previously unassigned tasks
      - value_priority = option to prioritise high value tasks 
                         (only effective when staff is a whole number)
      - group_priority = option to prioritise large groups tasks 
     
     output=
      - teams_tasks    = list of dictionaries, each characterising an allocatable team
      - left_tasks     = list of unallocatable tasks
    """

    if old:
        demands.extend(old)
    
    team_tasks = []
    left_tasks = []
    
    if isinstance(staff, list):
        # determine the staff available for each task value
        values = [stf['v'] for stf in staff]
        cond = np.in1d(np.arange(nvalues)+1, values)
        persons = np.zeros(nvalues, dtype=int)
        persons[cond] = [stf['s'] for stf in staff]
        
        # store available staff in local variables
        for p,per in enumerate(persons):
            exec("s{} = {}".format(p+1, per))
        
        tt = 1
        for v in range(nvalues):
            v_tasks = [dm for dm in demands if dm['v']==v+1]
            
            # prioritise larger groups assignment
            if group_priority:
                v_tasks = sorted(v_tasks, key=lambda n:n['s'], reverse=True)
            
            # assign each demanded task with the available staff
            for t,tk in enumerate(v_tasks):
                if eval('{} <= s{}'.format(tk['s'], v+1)):
                    Tdict = dict(T = 'T_'+str(tt), 
                                 A = 'A_'+str(tk['a']), 
                                 V = tk['v'], 
                                 S = tk['s'],
                                 W = tk['w']
                                )
                    tname = ('_').join([str(tk['a']),str(tk['k'])])
                    Tdict['K'] = 'K_'+tname
                    # update available staff for task value "v"
                    exec("s{} -= {}".format(v+1, tk['s']))
                    team_tasks.append(Tdict)
                    tt += 1
                else:
                    # if no more staff is available, store in left list
                    left_tasks.append(tk)
        
        return team_tasks, left_tasks
    else:
        # prioritise high value demanded tasks
        if value_priority:
            demands = sorted(demands, key=lambda n:n['v'], reverse=True)
        
        # prioritise larger groups assignment
        if group_priority:
            demands = sorted(demands, key=lambda n:n['s'], reverse=True)
        
        # assign each demanded task with the available staff
        tt = 1
        for d,dem in enumerate(demands):
            if dem['s'] < staff:
                Tdict = dict(T = tt, 
                             A = dem['a'], 
                             V = dem['v'], 
                             S = dem['s'], 
                             W = dem['w']
                            )
                tname = ('_').join([str(dem['a']),str(dem['k'])])
                Tdict['K'] = tname
                staff -= dem['s']
                team_tasks.append(Tdict)
                tt += 1
            else:
                left_tasks.append(dem)
        
        return team_tasks, left_tasks

###############################################################################

def knapSack(C, tv, ts, tm, tk, at, we, n):
    """
     Function solving the 0-1 Knapsack problem 
    
     input:
       - C  = capacity of the helicopter
       - tv = teams value array
       - ts = teams size array
       - tm = teams labels array
       - tk = teams tasks array
       - at = assets array
       - we = weather probability array
       - n  = number of teams to allocate
    
     output: a dictionary containing the following keys
       - 'v' = total value attained from allocation process
       - 'o' = total seats occupied on the helicopter
       - 'i' = indices of teams assigned to the helicopter
       - 't' = labels of teams assigned to the helicopter
       - 'k' = labels of tasks of teams assigned to the helicopter
       - 'a' = labels of assets of teams assigned to the helicopter
       - 'w' = weather probability of assets reached by helicopter
    
    """
    
    # initialiase an array of empty dictionaries
    Z = [[{'v':0, 'o':0, 'i':'', 't':'', 'k':'', 'a':'', 'w':0
          } for y in range(C+1)] for x in range(n+1)]

    # Build table Z[][] in bottom up manner 
    for i in range(n+1): 
        for s in range(C+1): 
            if i==0 or s==0: 
                Z[i][s]['v'] = 0
                Z[i][s]['o'] = 0
                Z[i][s]['w'] = 0
                Z[i][s]['i'] = ''
                Z[i][s]['t'] = ''
                Z[i][s]['k'] = ''
                Z[i][s]['a'] = ''
            elif ts[i-1] <= s:
                N1 = tv[i-1] 
                N2 = Z[i-1][s-ts[i-1]]['v']
                N3 = Z[i-1][s]['v']
                if N1+N2>=N3:
                    Z[i][s]['v'] = N1+N2
                    Z[i][s]['o'] = Z[i][s]['o'] + ts[i-1]
                    Z[i][s]['w'] = Z[i][s]['w'] + we[i-1]
                    Z[i][s]['i'] = Z[i][s]['i'] + '%i,'%(i-1)
                    Z[i][s]['t'] = Z[i][s]['t'] + '%s,'%tm[i-1]
                    Z[i][s]['k'] = Z[i][s]['k'] + '%s,'%tk[i-1]
                    Z[i][s]['a'] = Z[i][s]['a'] + '%s,'%at[i-1]
                    if N2!=0:
                        if Z[i][s]['a'] in Z[i-1][s-ts[i-1]]['a']:
                            Z[i][s]['w'] = Z[i-1][s-ts[i-1]]['w']
                        else:
                            Z[i][s]['w'] = Z[i][s]['w'] * Z[i-1][s-ts[i-1]]['w']
                        Z[i][s]['o'] = Z[i][s]['o'] + Z[i-1][s-ts[i-1]]['o']
                        Z[i][s]['i'] = Z[i][s]['i'] + Z[i-1][s-ts[i-1]]['i']
                        Z[i][s]['t'] = Z[i][s]['t'] + Z[i-1][s-ts[i-1]]['t']
                        Z[i][s]['k'] = Z[i][s]['k'] + Z[i-1][s-ts[i-1]]['k']
                        Z[i][s]['a'] = Z[i][s]['a'] + Z[i-1][s-ts[i-1]]['a']
                else:
                    Z[i][s]['v'] = N3
                    Z[i][s]['o'] = Z[i-1][s]['o']
                    Z[i][s]['w'] = Z[i-1][s]['w']
                    Z[i][s]['i'] = Z[i-1][s]['i']
                    Z[i][s]['t'] = Z[i-1][s]['t']
                    Z[i][s]['k'] = Z[i-1][s]['k']
                    Z[i][s]['a'] = Z[i-1][s]['a']
            else: 
                Z[i][s]['v'] = Z[i-1][s]['v']
                Z[i][s]['i'] = Z[i-1][s]['i']
                Z[i][s]['o'] = Z[i-1][s]['o']
                Z[i][s]['w'] = Z[i-1][s]['w']
                Z[i][s]['t'] = Z[i-1][s]['t']
                Z[i][s]['k'] = Z[i-1][s]['k']
                Z[i][s]['a'] = Z[i-1][s]['a']
    
    return Z[i][s]

###############################################################################

def permutations_no_repeat(array):
    """
     Function to calculate permutations without repetitions 
    
     input = an array or list with repeated elements
     output = permutations of the elements and their indices
    """
    
    Nperms = list(permutations(array))
    Iperms = list(permutations(range(len(array))))
    visited = set()

    Nunique = []
    Iunique = []

    i=0
    while (i < len(Nperms)):
        if Nperms[i] not in visited:
            visited.add(Nperms[i])
            Nunique.append(Nperms[i])
            Iunique.append(Iperms[i])
        i+=1
    
    return Nunique, Iunique

###############################################################################

def daily_teams_allocation(Helicopters, Hprobs, Teams):
    """
     Function to perform daily allocation of teams on available vehicles
    
     input: 
      - Helicopters = list of helicopters' capacity
      - Hprobs      = functionality probability of helicopters
      - Teams       = allocatable teams list 
                      (elements must be dictionaries characterising each team)
     
     output: 
      - dictionary of possible allocation patterns
    """
    
    assert all([isinstance(tm, dict) for tm in Teams]), "Not all dictionaries!"
    
    # calculate all possible helicopters' capacity orderings 
    Horderings, Hindxs = permutations_no_repeat(Helicopters)

    # build data arrays using the information in dictionaries
    tv = np.array([tm['V'] for tm in Teams])  # teams value array
    ts = np.array([tm['S'] for tm in Teams])  # teams staff array
    tk = np.array([tm['K'] for tm in Teams])  # team tasks array
    tm = np.array([tm['T'] for tm in Teams])  # team labels array
    at = np.array([tm['A'] for tm in Teams])  # assets labels array
    we = np.array([tm['W'] for tm in Teams])  # weather chances array
    nn = len(Teams)   # total number of daily allocatable teams 

    # initialise lists to contain results of all iterations 
    Varr, Oarr, Narr, Tarr = [], [], [], []
    Karr, Aarr, Harr, Parr = [], [], [], []

    for r,(Horder,Hind) in enumerate(zip(Horderings,Hindxs)):

        # initialise lists containing results of individual iteration
        V = 0    # cumulative allocation process value
        N = 0    # number of effectively allocated teams
        H, O, T, K, A, P = [], [], [], [], [], []

        # renaming of arrays (for reiteration purpose)
        newtv, newts, newtm, newtk = tv, ts, tm, tk
        newat, newwe, newn = at, we, nn

        # recursive Knapsack problem solution algorithm
        for h,ind in zip(Horder,Hind):
            # execution of knapSack function 
            res = knapSack(h, newtv, newts, newtm, newtk, newat, newwe, newn)

            # saving maximum value obtained
            V += res['v']
            
            # saving total occupied helicopter seats
            O.append(res['o'])
            
            # extracting the items allocated in last execution
            items = res['i'].split(',')
            items = [i for i in map(int,filter(bool,items))]
            # number of newly allocated teams
            N += len(items)
            # saving team labels allocated 
            teams = res['t'].split(',')
            teams = [t for t in filter(bool,teams)]
            T.append(teams)
            # save team tasks allocated 
            tasks = res['k'].split(',')
            tasks = [k for k in filter(bool,tasks)]
            K.append(tasks)
            # save team assets allocated
            assets = res['a'].split(',')
            assets = [a for a in filter(bool,assets)]
            A.append(assets)
            
            # save flight chances
            # product of weather probability x helicopter functionality
            W = res['w']
            P.append(W * Hprobs[ind])
            
            # save helicopter indices
            H.append(ind+1)

            # update operational arrays, removing allocated elements
            newtv = np.delete(newtv, items)
            newts = np.delete(newts, items)
            newtm = np.delete(newtm, items)
            newtk = np.delete(newtk, items)
            newat = np.delete(newat, items)
            newwe = np.delete(newwe, items)
            newn = len(newtm)

        # build global lists containing all results
        Varr.append(V)
        Harr.append(H)
        Narr.append(N)
        Oarr.append(O)
        Aarr.append(A)
        Parr.append(P)
        Tarr.append(T)
        Karr.append(K)

    Rdict = {'V' : Varr,  # allocatin pattern performance (APP) array
             'H' : Harr,  # patterns used helicopters indices array
             'N' : Narr,  # patterns allocated teams number array
             'A' : Aarr,  # patterns assets of teams array
             'O' : Oarr,  # patterns helicopters occupation array
             'T' : Tarr,  # patterns allocated teams label array
             'K' : Karr,  # patterns allocated teams task array 
             'P' : Parr}  # patterns flights probability array
    
    return Rdict

###############################################################################

def max_value_patterns(patterns):
    """
     Function to determine the maximum value achieved 
     in items allocation with Kanpsack recipe
     
     Return the correspoding patterns and value
    """
    
    maxV = max(patterns['V'])
    indm = np.where(np.array(patterns['V'])==maxV)[0]
    
    new_patterns = {}
    for k,v in patterns.items():
        new_patterns[k] = [v[i] for i in indm]
    
    return new_patterns, maxV

#------------------------------------------------------------------------------

def least_helicopters_patterns(patterns):
    """
     Function to determine the minimum number of helicopters used
     in items allocation with Kanpsack recipe
     
     Return the corresponding patterns and number of helicopters
    """
    
    hused = [len([l for l in pattern if l]) for pattern in patterns['A']]
    
    minH = min(hused)
    indm = np.where(np.array(hused)==minH)[0]
    
    new_patterns = {}
    for k,v in patterns.items():
        new_patterns[k] = [v[i] for i in indm]
    
    return new_patterns, minH

#------------------------------------------------------------------------------

def most_teams_patterns(patterns):
    """
     Function to determine the largest number of teams allocated
     in items allocation with Knapsack recipe
     
     Return the correspoding patterns and largest team number
    """
    
    maxN = max(patterns['N'])
    indm = np.where(np.array(patterns['N'])==maxN)[0]

    new_patterns = {}
    for k,v in patterns.items():
        new_patterns[k] = [v[i] for i in indm]
    
    return new_patterns, maxN

#------------------------------------------------------------------------------

def similar_destinations_patterns(patterns):
    """
     Function to count occurances of equal assets on the same helicopter
     and assign a positive score for each occurance
     
     Return highest ranking patterns and score achieved
    """
    
    scores = []
    for pattern in patterns['A']:
        Sa = 1
        for helicopter in pattern:
            if len(helicopter)>0:
                for k,d in Counter(helicopter).items():
                    if d>1:
                        Sa += d
        scores.append(Sa)
    
    maxS = max(scores)
    indm = np.where(np.array(scores)==maxS)[0]
    
    new_patterns = {}
    for k,v in patterns.items():
        new_patterns[k] = [v[i] for i in indm]
    
    return new_patterns, maxS

#------------------------------------------------------------------------------

def max_probability_patterns(patterns):
    """
     Function to find the pattern with the highest flights probability
     in items allocation with Knapsack recipe
     
     Return highest probability patterns and score 
    """
    
    # remove patterns with unused helicopters 
    # in case this has not been performed yet
    if len(patterns['A'])>1:
        patterns = least_helicopters_patterns(patterns)[0]
    
    maxP = max(patterns['P'])
    TrueFalse = np.array(patterns['P'])==maxP
    indm = np.where([all(l) for l in TrueFalse])[0]
    
    new_patterns = {}
    for k,v in patterns.items():
        new_patterns[k] = [v[i] for i in indm]
    
    scoreP = np.prod([p for p in maxP if p])
    return new_patterns, scoreP

###############################################################################

def random_assignment_score(asset_pattern):
    """
     Function to return the assets' allocation score
     for randomly assigned teams on helicopters
    """
    score = 1

    for pattern in asset_pattern:
        for k,d in Counter(pattern).items():
            if d>1:
                score += d
    return score

#------------------------------------------------------------------------------

def random_teams_allocator(Helicopters, Hfunction, Teams, iterations=10):
    """
     Function to randomly allocate teams onto the available helicopters
     
     input
      - Helicopters = list of helicopters' capacity
      - Hfunction = list of helicopters' functionality probability
      - Teams = list of dictionaries characterising each allocatable team
      - iterations = number of iterations to repeat (default=10)
      
     output
      - list of scores achieved
      - list of occupied helicopters
      - list of allocated teams per helicopter
      - list of asset destinations per helicopter
    
    """
    # list of helicopters' capacity index
    hhh = list(range(len(Helicopters)))

    # extract informations from Teams list of dictionaries
    iii = np.array(range(len(Teams)))
    TTT = np.array([tm['T'] for tm in Teams])
    VVV = np.array([tm['V'] for tm in Teams])
    SSS = np.array([tm['S'] for tm in Teams])
    AAA = np.array([tm['A'] for tm in Teams])
    WWW = np.array([tm['W'] for tm in Teams])

    # initialise empty list to store results
    Sarr, Harr, Tarr, Aarr = [], [], [], []
    
    for it in range(iterations):
        # get possible list of teams information lists
        tt = TTT[iii]
        vv = VVV[iii]
        ss = SSS[iii]
        aa = AAA[iii]
        ww = WWW[iii]

        # initialise iteration variables
        VV = 0
        PH = 1
        WW = []
        AT = []
        TM = []
        TT = set()
        HH = set()

        # helicopters assignment
        for h in hhh:
            C = Helicopters[h]
            at = []
            tm = []
            pw = 1
            for t,v,s,a,w in zip(tt,vv,ss,aa,ww):
                if t not in TT:
                    if C-s >= 0:
                        C -= s
                        VV += v
                        TT.add(t)
                        HH.add(h)
                        if a not in at:
                            pw *= w
                        at.append(a)
                        tm.append(t)

            PH *= Hfunction[h] * pw
            AT.append(at)
            TM.append(tm)

        score = random_assignment_score(AT)
        score *= VV * len(TT) * len(HH) * PH
        
        # store to final results lists    
        Sarr.append(score)
        Harr.append([Helicopters[k] for k in HH])
        Tarr.append(TM)
        Aarr.append(AT)
        
        # shuffle teams information and helicopters' indices order
        shuffle(hhh)
        shuffle(iii)
    
    return Sarr, Harr, Tarr, Aarr

import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


# a city represents a gene
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


#we treat the fitness as the inverse of the route distance. We want to minimize route distance, so a larger fitness score is better. 
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                #We must return to the starting city, so our total distance needs to be calculated accordingly
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    # tells us how good a route is (in terms of distance)
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    # we randomly select the order in which we visit each city
    route = random.sample(cityList, len(cityList))
    return route

# creates an ordered list with the route IDs and each associated fitness score.
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

# returns a list of route IDs, which we can use to create the mating pool in the matingPool function.
def selection(popRanked, eliteSize):
    selectionResults = []
    # we set up the roulette wheel by calculating a relative fitness weight for each individual.
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        # we pick a randomly number and compare it to fitness weights to select our mating pool
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    # extracting the selected individuals from our population.
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    # randomly select a subset of the first parent
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    #  fill the remainder of the route with the genes from the second parent in the order in which they appear,
    #  without duplicating any genes in the selected subset from the first parent
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def crossoverPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    # retain the best routes from the current population
    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        # fill out the rest of the next generation.
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# swap cities with a lower than mutationRate in order to get a new route 
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            # swap cities in order to not lose any of them
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen) # rank routes from current generation
    selectionResults = selection(popRanked, eliteSize) # determine potential parents 
    matingpool = matingPool(currentGen, selectionResults) # create the mating pool
    children = crossoverPopulation(matingpool, eliteSize) # crate the new generation
    nextGeneration = mutatePopulation(children, mutationRate) # apply mutation
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# create a population with as many routes as popSize
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def Distance(x1,y1,x2,y2):
    return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

def GeneratePoints(nr):
    return np.random.randint(1000, size=(nr, 2))


    
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    print("start")
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

def main():
        
    nr = 100
    cityList = []

    for i in range(0, nr):
        cityList.append(City(x=int(random.random() * 50), y=int(random.random() * 50)))


    # geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

    geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=2000)


main()
import networkx as nx
import random
import matplotlib.pyplot as plt
import itertools
import numpy as np
from timeit import default_timer as timer
from math import ceil, floor
from copy import deepcopy
import pandas as pd
from enum import Enum
import time
import pygame


# GAME
# =====
class Colors(Enum):
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)


class Game:
    def __init__(self, width, height, title):
        pygame.init()
        self.width, self.height = width, height
        self.title = title
        self.isRunning = True
        self.window = pygame.display.set_mode((self.width, self.height))

    def run(self):
        while self.isRunning:
            self.events_check()
            self.window.fill(Colors.white.value)
            # Render
            # ========
            # pygame.draw.circle(self.window, Colors.red.value, (self.width / 2, self.height / 2), 5)
            self.print_customers()
            # ========
            pygame.display.update()

    def events_check(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isRunning = False
                pygame.quit()

    def process_input(self):
        pass

    def print_customers(self):
        global agentList
        costumers = [agent for agent in agentList if (agent.__class__.__name__ == 'Customer')]
        for costumer in costumers:
            pos = (costumer.position[0] * self.width / 10, costumer.position[1] * self.height / 10)
            print(pos)
            pygame.draw.circle(self.window, Colors.red.value, pos, 10)


# GLOBAL VARIABLES
# =================
agentList = []
graphNodes = []
averageTime = []
graph = None


# GRAPH
# ======
def random_edge(graph, nb_edges, delete=True):
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    if delete:
        chosen_edges = random.sample(edges, nb_edges)
        for edge in chosen_edges:
            graph.remove_edge(edge[0], edge[1])
    else:
        chosen_nonedges = random.sample(nonedges, nb_edges)
        for nonedge in chosen_nonedges:
            graph.add_edge(nonedge[0], nonedge[1])


def generate_2d_graph(n, coef=False, delete=True, show=False):
    graph = nx.grid_2d_graph(n, n)
    if not coef:
        nb_ = int(len(list(graph.nodes)) * coef)
        random_edge(graph, nb_, delete)
    pos = nx.spring_layout(graph, iterations=100)
    graph.remove_edges_from(list(nx.isolates(graph)))
    graph = graph.to_directed()

    if show:
        nx.draw(graph, pos, node_color='b', node_size=20, with_labels=False)
        plt.title('Road Network')
        plt.show()

    return graph


# AGENT
# ======
class Agent:
    def __init__(self, _id, position, state):
        self.id = _id
        self.position = position
        self.state = state
        self.isAlive = True

    def to_string(self):
        pass

    def to_state(self):
        pass

    def update(self):
        pass

    def decide(self):
        pass


# CUSTOMER
# =========
class CustomerState(Enum):
    none = 1
    ordering = 2
    waiting = 3


class Customer(Agent):
    def __init__(self, _id, frequency):
        global graphNodes, agentList
        position = random.choice(graphNodes)
        super().__init__(_id, position, CustomerState.none)
        self.frequency = frequency
        self.iter = 0

    def pedido_arrived(self):
        self.state = CustomerState.none

    def update(self):
        if self.state == CustomerState.none:
            self.iter += 1

            if self.iter % self.frequency == 0:
                self.state = CustomerState.ordering

    def decide(self):
        if self.state == CustomerState.ordering:
            restaurants = [agent for agent in agentList if (agent.__class__.__name__ == 'Restaurant' and len(agent.preparingFood) < agent.capacity)]

            if len(restaurants) > 0:
                selectedRestaurant = random.choice(restaurants)

                print('Customer', self.id, 'choose restaurant', selectedRestaurant.id)

                timeToPrepare = random.randint(1, 3)
                food = Food(timeToPrepare, self)
                selectedRestaurant.addFood(food)
                self.state = CustomerState.waiting
            else:
                print('Customer', self.id, 'not found restaurants')

    def to_state(self):
        if self.state == CustomerState.waiting:
            print('I am customer', self.id, 'with state', self.state.name)

    def to_string(self):
        print('I am customer "', self.id, '" with state "', self.state.name, '" at the position "', self.position, '".')


# RESTAURANT
# ===========
class RestaurantState(Enum):
    waiting = 1
    preparing = 2


class Restaurant(Agent):
    def __init__(self, _id, capacity):
        global agentList, graphNodes
        restaurants_positions = [agent.position for agent in agentList if agent.__class__.__name__ == 'Restaurant']
        position = random.choice(graphNodes)
        while position in restaurants_positions:
            position = random.choice(graphNodes)
        super().__init__(_id, position, RestaurantState.waiting)

        self.capacity = capacity
        self.preparingFood = []
        self.readyFood = []
        self.deliveryFood = []

    def update(self):
        # Actualiza el estado de preparing Food
        for food in self.preparingFood:
            food.step()
        for food in self.readyFood:
            food.step()
        for food in self.deliveryFood:
            food.step()

    def decide_preparingfood(self):
        # Cambia el el estado de preparing a ready
        auxreadyFood = []

        for food in self.preparingFood:
            if food.ready_food():
                auxreadyFood.append(food)

        # eliminar y agregar food
        for food in auxreadyFood:
            self.preparingFood.remove(food)
            self.readyFood.append(food)

    def decide_callDistributor(self):
        # llamar a los delivery
        auxdeliveryFood = []

        for food in self.readyFood:
            distributors = [agent for agent in agentList if (agent.__class__.__name__ == 'Distributor'
                                                             and agent.state == DistributorState.waiting)]

            if (len(distributors) > 0) and (food not in auxdeliveryFood):
                selectedDistributor = random.choice(distributors)
                # cambio la ruta del distribuidor, agrego el food
                selectedDistributor.changeroute(self.position)
                selectedDistributor.state = DistributorState.getting
                food.targetdistributor = selectedDistributor
                auxdeliveryFood.append(food)

        # actualizar estado de food cambiados
        for food in auxdeliveryFood:
            self.readyFood.remove(food)
            self.deliveryFood.append(food)

    def decide_sendFood(self):
        # Verificar si el distribuidor está en la poicion restaurante
        distributors = [agent for agent in agentList if (agent.__class__.__name__ == 'Distributor'
                                                         and agent.state == DistributorState.getting
                                                         and agent.position == self.position)]

        sendingFood = list()
        for food in self.deliveryFood:
            for distributor in distributors:
                if food.targetdistributor == distributor:
                    # actualizar el distributor
                    sendingFood.append(food)
                    distributor.state = DistributorState.sending
                    distributor.changeroute(food.targetcustomer.position)
                    distributor.setFood(food)

        for food in sendingFood:
            self.deliveryFood.remove(food)

    def decide(self):
        if len(self.preparingFood) > 0:
            self.decide_preparingfood()
        if len(self.readyFood) > 0:
            self.decide_callDistributor()
        if len(self.deliveryFood) > 0:
            self.decide_sendFood()

    def addFood(self, food):
        self.preparingFood.append(food)

    def to_state(self):
        numberpreparing = len(self.preparingFood)
        if numberpreparing > 0:
            print('I am restaurant', self.id, 'preparing', numberpreparing, 'foods.')

            for food in self.preparingFood:
                print('Food to customer', food.targetcustomer.id, 'state', food.iter, ':', food.timetoprepare)

        numberready = len(self.readyFood)
        if numberready > 0:
            print('I am restaurant', self.id, 'ready', numberready, 'foods.')

        numberdelivery = len(self.deliveryFood)
        if numberdelivery > 0:
            print('I am restaurant', self.id, 'delivery', numberdelivery, 'foods.')

            for food in self.deliveryFood:
                print('Food to delivery', food.targetdistributor.id)

    def to_string(self):
        print('I am restaurant', self.id, 'with state', self.state.name, 'at the position ', self.position)


# DISTRIBUTOR
# ============
class DistributorState(Enum):
    waiting = 1
    getting = 2
    sending = 3


class Distributor(Agent):
    def __init__(self, id):
        global graph, graphNodes, averageTime
        self.setroute()
        super().__init__(id, self.position, DistributorState.waiting)
        self.food = None

    def update(self):
        if self.food is not None:
            self.food.step()

    def changeroute(self, target):
        # actualizar ruta del distribuidor
        n1 = self.position  # distributor
        n2 = target  # restaurant
        self.trip = nx.shortest_path(graph, n1, n2)

    def setroute(self, init=None):
        if init is None:
            self.position = random.choice(graphNodes)

        n1 = self.position
        n2 = random.choice(graphNodes)
        while n2 == n1:
            n2 = random.choice(graphNodes)
        self.trip = nx.shortest_path(graph, n1, n2)

    def decide(self):
        # si el distribuidor tiene pedido y ha llegado al destino
        if self.food is not None and self.food.targetcustomer.position == self.position:
            self.food.targetcustomer.pedido_arrived()
            self.state = DistributorState.waiting
            averageTime.append(self.food.iter)
            self.food = None

            # Si el distribuidor tiene ruta
        if len(self.trip) > 1:
            old = self.trip.pop(0)
            self.position = self.trip[0]
            # print('Distributor', self.id, 'moving from', old, 'to', self.current)

        # si ya no tengo ruta
        if len(self.trip) == 1:
            # crear un destino nuevo
            n1 = self.trip[0]
            n2 = random.choice(graphNodes)
            while n1 == n2:
                n2 = random.choice(graphNodes)
            trip = nx.shortest_path(graph, n1, n2)
            self.trip = trip

    def setFood(self, food):
        if self.food is None:
            self.food = food

    def to_string(self):
        pass

    def to_state(self):
        print('I am distributor', self.id, 'with state', self.state.name, 'and position', self.position, end='')
        print('my trip:', self.trip)


# FOOD
# =====
class Food:
    def __init__(self, timetoprepare, targetcustomer, targetdistributor=None):
        self.targetcustomer = targetcustomer
        self.timetoprepare = timetoprepare
        self.targetdistributor = targetdistributor
        self.iter = 0

    def ready_food(self):
        return self.iter >= self.timetoprepare

    def step(self):
        self.iter += 1


# APP
# ====
def timetostring(ticks):
    # 1 tick = 15 min
    # 4 ticks = 1 hora
    # 96 ticks = 24 hora
    dia = 0
    while ticks >= 96:
        ticks -= 96
        dia += 1

    hora = 0
    while ticks >= 4:
        ticks -= 4
        hora += 1

    minuto = 0
    while ticks >= 1:
        ticks -= 1
        minuto += 15

    return f"dia {dia} hora {hora} minuto {minuto}"


class App:
    def __init__(self, grid, coef, delete):
        global agentList, graph, graphNodes
        graph = generate_2d_graph(grid, coef, delete, show=False)
        graphNodes = list(graph.nodes)

    def addAgents(self, ncustomer, nrestaurants, ndistributor):
        for i in range(ncustomer):
            pos = random.choice(graphNodes)
            # ticks para que pida el cliente
            # 4 ticks = 1 hora
            # 32 ticks = 8 horas (3 veces al dia)
            # 48 tticks = 12 horas (2 veces al dia)
            # 144 ticks = 24 horas (1 vez al dia)

            frequency = random.choice([32, 48, 144])
            c = Customer(i, frequency)
            agentList.append(c)

        for i in range(nrestaurants):
            pos = random.choice(graphNodes)
            capacity = random.randint(10, 100)
            r = Restaurant(i, capacity)
            agentList.append(r)

        for i in range(ndistributor):
            d = Distributor(i)
            agentList.append(d)

    def initial_state(self):
        for ag in agentList:
            ag.to_string()

    def run(self, steps):
        for i in range(steps):
            time.sleep(0.25)
            print('-' * 30)
            print(timetostring(i))

            for a in agentList:
                a.decide()
            for a in agentList:
                a.update()

            for a in agentList:
                if a.isAlive == False:
                    agentList.remove(a)

            for a in agentList:
                a.to_state()


# MAIN
# =====
def main():
    global averageTime
    app = App(10, 0.2, True)
    app.addAgents(5, 10, 2)
    app.initial_state()

    app.run((12 * 60) // 15)
    print(averageTime)
    # print(sum(averageTime) / len(averageTime))

    game = Game(800, 600, "Hola mundo")
    game.run()


def main2():
    game = Game(800, 800, "Hola mundo")
    game.run()


if __name__ == '__main__':
    main()

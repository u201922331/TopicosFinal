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
import statistics
import matplotlib.pyplot as plt



class Colors(Enum):
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)

    customernone = (255, 170, 0)
    customerordering = (255, 97, 0)
    customerwaiting = (255, 97, 0)

    restaurantwaiting = (12, 255, 0)
    restaurantpreparing = (0, 255, 136)

    distributorwaiting = (0, 199, 255)
    distributorgetting = (0, 127, 255)
    distributorsending = (0, 50, 255)

class Game:
    def __init__(self, width, height, grid, title):
        pygame.init()
        self.width, self.height = width, height
        self.title = title
        self.isRunning = True
        self.window = pygame.display.set_mode((self.width, self.height))
        self.grid = grid
        self.gridvertical = self.height / (self.grid+2)
        self.gridhorizontal = self.width / (self.grid+2)

    def events_check(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isRunning = False
                pygame.quit()

    def process_input(self):
        pass
    
    def calculate_pos(self, pos):
        return (self.gridhorizontal + self.gridhorizontal * pos[0], self.gridvertical + self.gridvertical * pos[1])

    def print_system(self):
        global agentList, graph
        
        #Imprimir la grilla
        self.window.fill(Colors.white.value)
        for source, target in graph.edges:
            sourcepos = self.calculate_pos(source)
            targetpos = self.calculate_pos(target)
            pygame.draw.line(self.window, Colors.black.value, sourcepos, targetpos)

        for agent in agentList:
            pos = self.calculate_pos(agent.position)
            color = None
            if agent.__class__.__name__ == 'Customer':
                # print(pos)
                if agent.state == CustomerState.none:
                    color = Colors.customernone
                elif agent.state == CustomerState.ordering:
                    color = Colors.customerordering
                else:
                    color = Colors.customerwaiting

            elif agent.__class__.__name__ == 'Restaurant':
                if agent.state == RestaurantState.waiting:
                    color = Colors.restaurantwaiting
                else:
                    color = Colors.restaurantpreparing
            else:
                if agent.state == DistributorState.waiting:
                    color = Colors.distributorwaiting
                elif agent.state == DistributorState.sending:
                    color = Colors.distributorsending
                else:
                    color = Colors.distributorwaiting

            pygame.draw.circle(self.window, color.value, pos, 5)
                 

# GLOBAL VARIABLES
# =================
agentList = []
graphNodes = []

graph = []

#por cada restaurante o distribudor
food_list = [] #cantidad de tiempo del food
restaurant_list = [] #cantidad pedidos
distributor_list = [] #cantidad de pedidos

#por cada ejecucion del programa
food_timegeneral = [] #almacenar tiempo promedio
restaurant_general = [] #alamcenar promedio de capacidad
distributor_general = [] #media de pedidos que atendio

# GRAPH
# ======
def random_edge(nb_edges, delete=True):
    global graph
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    if delete:
        chosen_edges = random.sample(edges, nb_edges)
        for edge in chosen_edges:
            try:
                graph.remove_edge(edge[0], edge[1])
            except:
                print('Already removed')
            
    else:
        chosen_nonedges = random.sample(nonedges, nb_edges)
        for nonedge in chosen_nonedges:
            try:
                graph.add_edge(nonedge[0], nonedge[1])
            except:
                print('Already added')

def generate_2d_graph(n, coef=False, delete=True, show=False):
    global graph
    # print('before generate', len(graph.nodes))
    graph = nx.grid_2d_graph(n, n)
    print('already generate', len(graph.nodes))

    if coef is not False:
        nb_ = int(len(list(graph.nodes)) * coef)
        random_edge(nb_, delete)
        print('nb', nb_)
    
    print('already generate', len(graph))

    
    graph.remove_edges_from(list(nx.isolates(graph)))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    pos = nx.spring_layout(graph, iterations=100)

    graph = graph.to_directed()
    print('afeter generate', len(graph.nodes))

    if show:
        nx.draw(graph, pos, node_color='b', node_size=20, with_labels=False)
        plt.title('Road Network')
        plt.show()

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
        customerspositions = [agent.position for agent in agentList if agent.__class__.__name__=='Customer']
        position = random.choice(graphNodes)
        while position in customerspositions:
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
        print('I am customer "', self.id, '" with state "', self.state.name, '" at the position "', self.position, 'frequency', self.frequency, 'iterador', self.iter)


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
        self.numberpreparedfood = 0
        self.maxpreparedfood = 0

    def update(self):
        # Actualiza el estado de preparing Food
        for food in self.preparingFood:
            food.step()
        self.maxpreparedfood = max(self.maxpreparedfood, len(self.preparingFood))

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
            self.numberpreparedfood += 1

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
        print('I am restaurant', self.id, 'with state', self.state.name, 'at the position ', self.position, 'capacity', self.capacity)
        print('Preparing food', self.preparingFood)
        print('Ready food', self.readyFood)
        print('deliveryFood', self.deliveryFood)
        print('numberpreparedfood', self.numberpreparedfood)
        print('maxpreparedfood', self.maxpreparedfood)
    
    def print_capacity(self):
        return self.maxpreparedfood / self.capacity

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
        self.numberdeliveryfood = 0

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
        global food_list
        # si el distribuidor tiene pedido y ha llegado al destino
        if self.food is not None and self.food.targetcustomer.position == self.position:
            self.food.targetcustomer.pedido_arrived()
            self.state = DistributorState.waiting
            food_list.append(self.food.iter)
            self.numberdeliveryfood += 1
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
        print('I am distributor', self.id, 'with state', self.state.name, 'and position', self.position, end='')
        print('my trip:', self.trip)

    def to_state(self):
        print('I am distributor', self.id, 'with state', self.state.name, 'and position', self.position, end='')
        print('Food', self.food)
        print('numberdeliveryfood', self.numberdeliveryfood)

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
        generate_2d_graph(grid, coef, delete, show=False)
        graphNodes = list(graph.nodes)
        self.grid = grid
        print(graph.nodes)
        print(graph.edges)


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

        game = Game(800, 600, self.grid, "Hola mundo")
        
        for i in range(steps):
            #time.sleep(1.0)
            # Render pygame
            #game.events_check()
            #game.print_system()
            #pygame.display.update()


            print('-' * 30)
            print(timetostring(i))

            for a in agentList:
                a.decide()
            for a in agentList:
                a.update()

            for a in agentList:
                if a.isAlive == False:
                    agentList.remove(a)

            # for a in agentList:
            #     a.to_state()

        self.time_food()
        self.time_restaurant()
        self.time_distributor()

    def time_food(self):
        global food_list, food_timegeneral
        #print('food_list', food_list)
        if len(food_list) > 0:
            mean = statistics.mean(food_list)
        else:
            mean = 0
        #print('mean_food_list', mean)
        food_timegeneral.append(mean)



    def time_restaurant(self):
        global restaurant_list, restaurant_general
        restaurant_list =  [agent.print_capacity() for agent in agentList if agent.__class__.__name__ == 'Restaurant']
        #print('restaurant_list', restaurant_list)
        print(restaurant_list)
        # mean = statistics.mean(restaurant_list)
        restaurant_general.append(restaurant_list)

    def time_distributor(self):
        global distributor_list, distributor_general
        distributor_list =  [agent.numberdeliveryfood for agent in agentList if agent.__class__.__name__ == 'Distributor']
        #print('distributor_list', distributor_list)
        # mean = statistics.mean(distributor_list)
        distributor_general.append(distributor_list)

def print_general_status(steps, nrestaurants, ndistributors):
    global food_timegeneral, restaurant_general, distributor_general
    print('General Status:')
    print('food_timegeneral', food_timegeneral)
    print()
    print('restaurant_general', restaurant_general)
    print()
    print('distributor_general', distributor_general)

    #Creates just a figure and only one subplot
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    #Food
    axes[0].set_title('Tiempo promedio que demora el pedido por ejecución')
    x = np.arange(steps)
    axes[0].plot(x, food_timegeneral)
    axes[0].set_yticks(np.arange(0, max(food_timegeneral)+1, 1))

    #restaurant
    axes[1].set_title('Porcentaje logrado de capacidad por restaurante en cada ejecución')
    x = np.arange(nrestaurants)
    for timerestaurant in restaurant_general:
        axes[1].plot(x, timerestaurant)
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].set_xticks(x)

    #distributor
    axes[2].set_title('Cantidad de delivery por motoizado en cada ejecución')
    x = np.arange(ndistributors)
    max_delivery = 0
    for distributor in distributor_general:
        axes[2].plot(x, distributor)
        max_delivery = max(max_delivery, max(distributor))
        print('max_delivery', max_delivery)
        print('max_delivery', distributor)
        
    axes[2].set_yticks(np.arange(0, max_delivery+1))
    axes[2].set_xticks(x)

    fig.tight_layout()
    plt.show()



# MAIN
# =====
def main():
    global food_list, restaurant_list, distributor_list, agentList, graph, graphNodes
    
    ncustomers = 5
    nrestaurants = 5
    ndistributors = 2
    steps = 1000
    for _ in range(steps): #10000 veces correr el programa
        food_list = []
        restaurant_list = []
        distributor_list = []
        agentList = []
        graph = []
        graphNodes = []
        app = App(3, 0.3, True)    
        app.addAgents(ncustomers, nrestaurants, ndistributors)
        app.initial_state()
        app.run((25 * 60) // 15)
        # print('='*30)
        # print('Genral status')
        # print('averagefood', averagefood)
        # print('averagerestaurant', averagerestaurant)
        # print('averagedistributor', averagedistributor)
    
    #Imprimir gráficos
    print_general_status(steps, nrestaurants,ndistributors )
    #print(average_list)

    # print(sum(averageTime) / len(averageTime))    

if __name__ == '__main__':
    main()

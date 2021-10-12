import math
import random
from matplotlib import pyplot as plt


############## les bibliotheques de l'interface graphique ##################

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton , QFileDialog , QLineEdit,  QInputDialog,QTextEdit, QWidget, QPushButton, QVBoxLayout, QHBoxLayout 
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5 import QtCore
from interface import *
from pathlib import Path

 

#############################################################################


########################### Ant Colony optimization #####################################
class SolveTSPUsingACO: #class mere
    
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges): #constructeur de fourmi
            self.alpha = alpha
            self.beta = beta
            # α and β sont deux paramètres positifs réglables qui contrôlent les poids relatifs de la trace des phéromones et de la visibilité heuristique
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None        
            self.distance = 0.0

            """Supposons que vous ayez le choix entre 10 éléments et que vous choisissez en générant un nombre aléatoire compris entre 0 et 1.
            Vous divisez la plage de 0 à 1 en dix segments qui ne se chevauchent pas,
            chacun proportionnel à la forme physique de l'un des dix éléments"""

        def _select_node(self): #fonction utilisé par la fourmis en chaque noeud pour choisir le noeud suivant
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            #choisir un float aleatoire entre 0 et la valeur de roulette wheel
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self): #retourne la liste des noeuds visités en ordre
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self): # retourne la distance de la tour
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, colony_size=10, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
       
        self.colony_size = colony_size #nombre total des fourmis
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps #nombre des pas fait par la fourmi
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf") #float("inf") = l'infinie

    def _add_pheromone(self, tour, distance, weight=1.0): #ajout de pheremone dans chaque noeud
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += pheromone_to_add

    def _acs(self): #mise a jour de pheremone apres chaque tour + determination de "best_distance" et "best_tour"
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)
################################################################################################




                    
####################################  Resultat  ####################################################

    def run(self):
        print('Started : ACS')
        self._acs()

        print('Ended : ACS')
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))
############################################################################################################




        ############################### GRAPH ##########################################

        
    def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title('ACS')
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        if save:
            if name is None:
                name = 'ACS.png'
            plt.savefig(name, dpi=dpi)
        plt.show()
        plt.gcf().clear()
        ##################################################################################


####################################### Main ###########################################       

def main():
    _colony_size = 5
    _steps = 50
    cities = []
    _nodes = []
    with open('./data/villes.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities.append(dict(index=int(city[0]), x=int(city[1]), y=int(city[2])))
            _nodes.append((int(city[1]), int(city[2])))
         
    acs = SolveTSPUsingACO(colony_size=_colony_size, steps=_steps, nodes=_nodes)
    acs.run()
    acs.plot()
    
#######################################################################################


######################################### INTERFACE GRAPHIQUE #################################################
class myApp(Ui_Dialog):
    def __init__(self,window):
        self.setupUi(window)
        self.pushButton.clicked.connect(self.Button1Action)
        self.pushButton_2.clicked.connect(self.Button2Action)
        
       
 
    def Button1Action(self):
         main()
    def Button2Action(self):
                                                
        self.w = Notepad()
        self.w.show()
        
class Notepad(QWidget):

    def __init__(self):
        super(Notepad, self).__init__()
        self.text = QTextEdit(self)
        self.clr_btn = QPushButton('Clear')
        self.sav_btn = QPushButton('Save')
        self.opn_btn = QPushButton('Open')

        self.init_ui()

    def init_ui(self):
        v_layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        h_layout.addWidget(self.clr_btn)
        h_layout.addWidget(self.sav_btn)
        h_layout.addWidget(self.opn_btn)

        v_layout.addWidget(self.text)
        v_layout.addLayout(h_layout)

        self.sav_btn.clicked.connect(self.save_text)
        self.clr_btn.clicked.connect(self.clear_text)
        self.opn_btn.clicked.connect(self.open_text)

        self.setLayout(v_layout)
        self.setWindowTitle('Modifier les coordonnées des villes')

        self.show()

    def save_text(self):
        filename = QFileDialog.getSaveFileName(self, 'Save File', str(Path.home()))
        with open(filename[0], 'w') as f:
            my_text = self.text.toPlainText()
            f.write(my_text)

    def open_text(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', str(Path.home()))
        with open(filename[0], 'r') as f:
            file_text = f.read()
            self.text.setText(file_text)

    def clear_text(self):
        self.text.clear()
    
         
    


################################################################################################################   


######### Programme principal #############

         
if __name__ == "__main__":  
    App = QApplication(sys.argv)
    window = QMainWindow()
    ui= myApp(window)
    window.show()
    App.exec_()


############################################




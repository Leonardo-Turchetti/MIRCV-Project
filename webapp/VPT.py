import random
import math
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Point:
  def __init__(self, features, img_id, label):
    self.features = features
    self.img_id = img_id
    self.label = label

class Node:
  def __init__(self):
    self.leaf = False
    self.left = None
    self.right = None
    self.median = None
    self.mean = None
    self.pivot = None

def distanceOfTwoPoints(point1,point2,metric=""):
  if metric.lower() == "euclidean":
    result = np.sqrt(np.sum(np.square(point1.features-point2.features)))
  elif metric.lower() == "manhattan":
    result = np.sum(abs(point1.features-point2.features))
  elif metric.lower() == "cosine":
    A = np.asarray(point1.features).reshape(1,-1)
    B = np.asarray(point2.features).reshape(1,-1)
    #Thanks to squeeze(), we return, for istance, 0.7849372 shape () instead of a [[0.7849372]] with (1,1) of shape
    result = 1 - cosine_similarity(A,B).squeeze()  
  else:
    print("ERRORE: Metrica non riconosciuta!")
    return 0
  return result

class Stack:
  # A list of (distance_i,class point_i)
  def __init__(self,k):
    self.size = k
    self.data = []
    self.index = 0
  
  def addElement(self,distance,point):

    if(self.isFull()):
      self.deleteLastElement()

    elem = (distance,point)
    self.data.append(elem)
    self.data.sort(key=lambda y: y[0])
    

      
  def deleteLastElement(self):
    if not self.isFull():
      print("Errore: Lista deve essere piena")
    else:
      del self.data[-1]
  
  #Pay attention in knn searching, dnn indicates the d_max of a stack
  # [(3,p4),(5,p2),(12,p5)]
  # d_max is equal to 12
  def update_d_max(self): #update dnn
    return self.data[-1][0]

  def isFull(self):
    return len(self.data) == self.size

class VPTree:

  def __init__(self, feature_schemas, bucket_size, distanceMetric):
    self.bucket_size = bucket_size
    self.feature_schemas = feature_schemas.copy()
    self.nodes = 0
    self.internalNodes = 0
    self.leafNodes = 0
    self.root = Node()
    self.totalFeatures = len(feature_schemas)
    self.distanceMetric = distanceMetric
    self.build(self.root, self.feature_schemas)


  def build(self, node, feature_subset):
    pivot = random.choice(feature_subset)
    feature_subset.remove(pivot) 

    #Nel calcolo delle distance devo escludere il vantage point
    distances = []

    #Compute distances in according to chosen pivot 
    for feature in feature_subset:   
      dist = distanceOfTwoPoints(feature,pivot,self.distanceMetric)
      distances.append(abs(dist))

    distances = np.array(distances)
    median = np.median(distances)

    subset1 = []
    subset2 = []

    #Split the remaining features in accoring to computed median
    for feature in feature_subset:
      dist = distanceOfTwoPoints(feature,pivot,self.distanceMetric)
      dist = abs(dist)
      if dist <= median:
        subset1.append(feature)
      else:
        subset2.append(feature)

    node.median = median
    node.pivot = pivot
    node.left = Node()
    node.right = Node()
    self.internalNodes += 1
    self.nodes += 1

    if len(subset1) <= self.bucket_size:
      node.left.subset = subset1
      node.left.leaf = True
      self.leafNodes += 1
      self.nodes += 1
    else:
      self.build(node.left, subset1)

    if len(subset2) <= self.bucket_size:
      node.right.subset = subset2
      node.right.leaf = True
      self.leafNodes += 1
      self.nodes += 1
    else:
      self.build(node.right, subset2)
  
  def range_search(self,query,range):
    self.feature_list = []
    self.recursive_range_search(self.root,query,range)
    return self.feature_list

  def recursive_range_search(self,node,query,range):
    if node.leaf == True:
      
      for point in node.subset:
        dist = distanceOfTwoPoints(query,point,self.distanceMetric)
        dist = abs(dist)
        if(dist <= range):
          tmp = []
          tmp.append(point)
          tmp.append(dist)
          self.feature_list.append(tmp)
      return
    
    #We insert the pivot if it is near to the query
    dist = distanceOfTwoPoints(query,node.pivot,self.distanceMetric)
    dist = abs(dist)
    if(dist <= range):
      tmp = []
      tmp.append(node.pivot)
      tmp.append(dist)
      self.feature_list.append(tmp)


    if(dist - range <= node.median):
      self.recursive_range_search(node.left, query, range)
    if(dist + range >= node.median):
      self.recursive_range_search(node.right, query, range)
    return

  def knn(self, query, k):
    #global time
    start_time = time.time()
    self.knn = Stack(k)
    self.d_max = math.inf
    self.visited = 0
    self.recursive_knn(self.root, query)
    results = []

    for elem in self.knn.data:
      results.append((elem[0],elem[1])) 

    self.timeknn = time.time() - start_time
    print("Time consumed: ", self.timeknn)
    print("Visited/Total elements: ",self.visited,"/",self.totalFeatures)

    return results
  
  def recursive_knn(self,node,query):
    if node.leaf == True:
      for point in node.subset:
        self.visited += 1
        #print("--foglia")
        #print("id node: ",point.img_id)
        #print("features node: ",point.features)
        distance = distanceOfTwoPoints(query,point,self.distanceMetric)
        #distance = query.features - point.features                    <------
        distance = abs(distance)
        #print("distance: ",distance)
        #print("actual dnn: ",self.d_max)
        #print("actual knn: ",self.knn.data)
        if not self.knn.isFull():
          self.knn.addElement(distance,point)
          self.d_max = self.knn.update_d_max()
        elif distance < self.d_max:
          self.knn.addElement(distance,point)
          #remove last element inside the stack
          self.d_max = self.knn.update_d_max()
        
        #print("new dnn: ",self.d_max)
        #print("new knn: ",self.knn.data)
        #print("-----------------------")
      return

    self.visited += 1
    #print("id node: ",node.pivot.img_id)
    #print("features node: ",node.pivot.features)
    #distance = query.features - node.pivot.features                  <-------
    distance = distanceOfTwoPoints(query,node.pivot,self.distanceMetric)
    distance = abs(distance)
    #print("distance: ",distance)
    #print("actual dnn: ",self.d_max)
    #print("actual knn: ",self.knn.data)

    if not self.knn.isFull():
      self.knn.addElement(distance,node.pivot)
      self.d_max = self.knn.update_d_max()
    elif distance < self.d_max:
      self.knn.addElement(distance,node.pivot)
      #remove last element inside the stack
      self.d_max = self.knn.update_d_max()
    #print("new dnn: ",self.d_max)
    #print("new knn: ",self.knn.data)

    if distance - self.d_max <= node.median:
      self.recursive_knn(node.left, query)
    if distance + self.d_max >= node.median:
      self.recursive_knn(node.right, query)
    return

  # function to print the tree, it will call print_tree
  def print_root(self):
      self.print_tree(self.root)

  def print_number_nodes(self):
    print("Total nodes: ", self.nodes)
    print(self.internalNodes," are Internal Nodes")
    print(self.leafNodes," are Leaf Nodes")
  # recursive function to print the tree
  def print_tree(self, node):
      if node.leaf == True:
          print("LEAF")
          print("Number of points in the node: ",len(node.subset))
          for elem in node.subset:
            print(elem.img_id)
            print(elem.features)
            print(elem.label)
          return
      else:
          print("INTERNAL")
          print("Pivot: ", node.pivot.img_id)
          print("Median: ", node.median)
          print("left")
          self.print_tree(node.left)
          print("right")
          self.print_tree(node.right)
          return
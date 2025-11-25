import pymunk
import pygame
import random
print("Pymunk version:", pymunk.version)
#FPS
FPS=60

# create space
s = pymunk.Space()

# create the body for the sprite
b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
b.position = (100, 100)
b.velocity = (50, 20)

#create a list of the bodies and shapes  besides the sprite
bodies=[]
shapes=[]
shapeDict={}

# create the sprite shape
c = pymunk.Circle(b, 10)
c.collision_type=1
c.sensor = True
c.collision_type = 1
s.add(b, c)

# pygame setup
global width
width=800

global height
height=800

pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
running = True

#create some examplery polygons
triangle = [
    (0, 0),
    (60, 0),
    (30, 50)
]
pentagon = [
    (0, 30),
    (25, 0),
    (60, 15),
    (55, 50),
    (15, 60)
]
quad = [
    (0, 0),
    (70, 10),
    (50, 60),
    (10, 50)
]
hexagon = [
    (20, 0),
    (60, 0),
    (80, 30),
    (60, 60),
    (20, 60),
    (0, 30)
]

polyList=[triangle,pentagon,quad,hexagon]


for i in range(4):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = (random.randint(100, 700), random.randint(100, 700))
    body.velocity = (random.randint(5, 200), random.randint(5, 200))

    shape = pymunk.Poly(body, polyList[i])
    shape.sensor=False

    s.add(body, shape)

    bodies.append(body)
    shapes.append(shape)



#creating random polygons

def randomPolygon():
  lip=random.randint(1,4)#which lip should the polygon spawn near
  ver=random.randint(2,5)#randomly choose the amount of vertices of the polygon
  maxX=0
  maxY=0
  minX=100#max x size
  minY=100#max y size
  p1=(maxX,0)
  p2=(0,maxY)
  p3=(minX,0)
  p4=(0,maxY)
  offsetPoint=(0,0)
 
  MARGIN=50
  
  
    
    
    
  points=[]
  points.append(offsetPoint)#body positioning will be relative to (0,0)
  for i in range(ver+1):
    p=(random.randint(0,minX),random.randint(0,minY))
    points.append(p)
    if p[0]>p1[0]:
      p1=p
    if p[0]<p3[0]:
      p3=p
    if p[1]>p1[1]:
      p1=p
    if p[1]<p3[1]:
      p3=p
   
  body=pymunk.Body(body_type=pymunk.Body.KINEMATIC)
  shape = pymunk.Poly(body, points)


  if lip == 1:# spawn to the RIGHT of screen, move left
        body.position = (width + MARGIN - p3[0], random.randint(0, height))
        body.velocity = (random.randint(-100, -20), random.randint(-40, 40))
  if lip == 2:# spawn ABOVE screen, move down
        body.position = (random.randint(0, width), -MARGIN - p2[1])
        body.velocity = (random.randint(-40, 40), random.randint(50, 200))
  if lip == 3:# spawn to the LEFT of screen, move right
        body.position = (-MARGIN - p1[1], random.randint(0, height))
        body.velocity = (random.randint(20, 100), random.randint(-40, 40))
  if lip==4:# lip == 4, spawn BELOW screen, move up
        body.position = (random.randint(0, width), height + MARGIN - p4[1])
        body.velocity = (random.randint(-40, 40), random.randint(-200, -50))

  shape.collision_type=2

  return (body,shape)
 
 

def IsPolyInBoundaries(shape):#checks if a polygon is in bounds
 
  points = [shape.body.local_to_world(v) for v in shape.get_vertices()]
  for point in points:
    if (point[0]>0 and point[0]<800) and (point[1]>0 and point[1]<800):
      return True
    
  return False
  
def polyInbounds(shape):#checks if a polygon is out of bounds,but specificly after it was alrdy in the bounds
   if (shape in shapeDict)==False and IsPolyInBoundaries(shape)==False:
    shapeDict[shape]=True
   
   points = [shape.body.local_to_world(v) for v in shape.get_vertices()]
   for point in points:
        if (point[0]>0 and point[0]<800) and (point[1]>0 and point[1]<800):
          return True
    
   shapeDict.pop(shape) 
   return False
    



#the pygame setup
dt=1/FPS
frame=0

#test
bod, shape = randomPolygon()

s.add(bod,shape)
shapes.append(shape)
bodies.append(bod)



spawnPoly=1#controls when to spawn polygons randomly

#add a handler and a begin function for collisions and handling them 


def on_begin(arbiter, space, data):
   print("hi")
   return True





while running:
    frame+=1#keep track of the frames
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("purple")

    # step physics
   
    s.step(dt)#60 steps per frame

    if spawnPoly==frame:#controls when the polygons are created randomly
      spawnPoly=random.randint(1,60)+frame
      bod, shape = randomPolygon()

      s.add(bod,shape)
      shapes.append(shape)
      bodies.append(bod)



    for shape in shapes:
     
     points = [shape.body.local_to_world(v) for v in shape.get_vertices()]
     points = [(int(p.x), int(p.y)) for p in points]
     pygame.draw.polygon(screen, "red", points)
     if polyInbounds(shape)==False:
       s.remove(shape,shape.body)
       shapes.remove(shape)
       bodies.remove(shape.body)



    # draw body
    pygame.draw.circle(
        screen,
        "red",
        (int(b.position.x), int(b.position.y)),
        40
    )

    
    keys = pygame.key.get_pressed()
    move = pymunk.Vec2d(0, 0)
    if keys[pygame.K_w]:
     move += (0, -300 * dt)
    if keys[pygame.K_s]:
     move += (0, 300 * dt)
    if keys[pygame.K_a]:
     move += (-300 * dt, 0)
    if keys[pygame.K_d]:
     move += (300 * dt, 0)

    b.position += move


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()


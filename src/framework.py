import pymunk
import pygame
import random
import neat
import os
import math
import pickle

# --- CONSTANTS ---
FPS = 60
WIDTH, HEIGHT = 800, 800
SHOW_GRAPHICS = False 
NUM_RAYS = 8          
SENSE_DIST = 350      # Increased so bot sees danger earlier

def get_inputs(bot_body, bot_vel, hazards, space):
    # boundary distance inputs (normalized 0 to 1)
    # 0.0 means touching the wall, 1.0 means at the far opposite side
    dist_left = bot_body.position.x / WIDTH
    dist_right = (WIDTH - bot_body.position.x) / WIDTH
    dist_top = bot_body.position.y / HEIGHT
    dist_bottom = (HEIGHT - bot_body.position.y) / HEIGHT
    
    wall_data = [dist_left, dist_right, dist_top, dist_bottom]
    # 1. raycasting (8 inputs)
    sensors = []
    for i in range(NUM_RAYS):
        angle = (math.pi * 2 / NUM_RAYS) * i
        ray_vec = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        start = bot_body.position
        end = start + ray_vec * SENSE_DIST
        info = space.segment_query_first(start, end, 1, pymunk.ShapeFilter())
        sensors.append(info.alpha if info else 1.0)
    
    # 2. sort hazards and prepare relative data
    hazards.sort(key=lambda h: (h.body.position - bot_body.position).length)
    
    h_data = [0.0] * 8 
    dot_products = [0.0, 0.0]
    pincer_risk = 0.0
    rel_vectors = []

    for i in range(min(len(hazards), 2)):
        h = hazards[i]
        # Relative Vector normalized by SENSE_DIST
        rel = (h.body.position - bot_body.position) / SENSE_DIST
        rel_vectors.append(rel)
        
        idx = i * 4
        h_data[idx] = max(-1.0, min(1.0, rel.x))
        h_data[idx+1] = max(-1.0, min(1.0, rel.y))
        h_data[idx+2] = h.body.velocity.x / 200
        h_data[idx+3] = h.body.velocity.y / 200
        
        # 3. Dot Product: Directional Danger
        if bot_vel.length > 0:
            # Normalized bot_vel dot relative_pos_to_hazard
            dot_products[i] = bot_vel.normalized().dot(rel.normalized())

    # 4. cross Product-are hazards pinning the bot?
    if len(rel_vectors) == 2:
        pincer_risk = abs(rel_vectors[0].cross(rel_vectors[1]))

    # 5. Normalized Bot Position
    bot_pos_data = [bot_body.position.x / WIDTH, bot_body.position.y / HEIGHT]

    # total =4(distance from walls) + 8 (rays) + 8 (hazards) + 2 (pos) + 2 (dots) + 1 (cross,pincer) = 25
    return sensors + h_data + bot_pos_data + dot_products + [pincer_risk]+wall_data

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        
        s = pymunk.Space()
        
        # Static walls so raycasts "see" the edges
        static_lines = [
            pymunk.Segment(s.static_body, (0, 0), (WIDTH, 0), 1),
            pymunk.Segment(s.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1),
            pymunk.Segment(s.static_body, (0, 0), (0, HEIGHT), 1),
            pymunk.Segment(s.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 1)
        ]
        for line in static_lines:
            line.sensor = True
            s.add(line)

        b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        b.position = (WIDTH/2, HEIGHT/2)
        c = pymunk.Circle(b, 12)
        s.add(b, c)
        
        hazards = []
        running = True
        spawn_timer = 0
        dt = 1/FPS
        
        # Tracking for Jitter and Rewards
        current_vel = pymunk.Vec2d(0, 0)
        prev_dist_to_hazards = {} # To track if we are getting further away

        for frame_count in range(5000):
            if not running: break
            
            # 1. GET INPUTS & THINK
            inputs = get_inputs(b, current_vel, hazards, s)
            output = net.activate(inputs)
            
            # 2. DIRECT VELOCITY OUTPUT (2 outputs from -1 to 1)
            target_vx = output[0] * 450 
            target_vy = output[1] * 450
            new_vel = pymunk.Vec2d(target_vx, target_vy)
            
            # 3. JITTER PENALTY (check change in velocity)
            accel = (new_vel - current_vel).length
            genome.fitness -= (accel / 500) * 0.1 # punish jerky changes

            # Update state
            current_vel = pymunk.Vec2d(target_vx, target_vy)
            b.position += current_vel * dt

            #3.2 getting too close to the walls penalty
            if b.position.x<40.0 or b.position.x>(WIDTH-40.0) or b.position.y<40.0 or b.position.y>(HEIGHT-40.0):
                genome.fitness-=1.2

            # 4. BOUNDARY & CENTER LOGIC
            b.position = (max(20, min(WIDTH-20, b.position.x)), max(20, min(HEIGHT-20, b.position.y)))
            
            # survival reward
            genome.fitness += 1

            # 1. MOVEMENT REWARD: reward velocity, but not jitter
            # This stops the bot  from sitting in the center.
            if current_vel.length > 50:
                genome.fitness += 0.2

            # 3. PROXIMITY PENALTY & DODGE REWARD
            if hazards:
                total_proximity_pressure = 0
                for i, h in enumerate(hazards[:3]): # Check closest 3
                    dist = (h.body.position - b.position).length
                    
                    
                    # if within 120, lose fitness exponentialy by adding up the proximity losses
                    if dist < 120:
                        #A. the closer it is, the more fitness we lose
                        proximity_loss = (120 - dist) / 100
                        total_proximity_pressure += proximity_loss
                        genome.fitness -= total_proximity_pressure * 2.0#the more near object there are, the more fitness we lose

                    # B. the dodge reward
                    # only reward moving away if the object is actually a threat (< 200)
                    if i in prev_dist_to_hazards and dist < 200:
                        if dist > prev_dist_to_hazards[i] + 0.2:
                            # big reward for active dodging
                            genome.fitness += 2
                    
                    prev_dist_to_hazards[i] = dist
            # --- UPDATED SPAWNING (Fix the blind spots) ---
            if spawn_timer <= 0:
                spawn_timer = random.randint(20, 40)
                h_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    h_body.position = (random.randint(0, WIDTH), -50)
                    h_body.velocity = (random.randint(-120, 120), random.randint(60, 180))
                elif side == 'bottom':
                    h_body.position = (random.randint(0, WIDTH), HEIGHT + 50)
                    h_body.velocity = (random.randint(-120, 120), random.randint(-180, -60))
                elif side == 'left':
                    h_body.position = (-50, random.randint(0, HEIGHT))
                    h_body.velocity = (random.randint(60, 180), random.randint(-120, 120))
                else: # Right side
                    h_body.position = (WIDTH + 50, random.randint(0, HEIGHT))
                    h_body.velocity = (random.randint(-180, -60), random.randint(-120, 120))

                h_shape = pymunk.Poly.create_box(h_body, (40, 40))
                s.add(h_body, h_shape)
                hazards.append(h_shape)
            spawn_timer -= 1

            s.step(dt)
            
            # 7. COLLISIONS & CLEANUP
            for shape in hazards[:]:
                shape.body.position += shape.body.velocity * dt
                if shape.point_query(b.position).distance < 12:
                    genome.fitness -= 400 
                    running = False
                if (shape.body.position.x < -100 or shape.body.position.x > WIDTH+100 or 
                    shape.body.position.y < -100 or shape.body.position.y > HEIGHT+100):
                    s.remove(shape.body, shape)
                    hazards.remove(shape)

def run_training(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    
    winner = p.run(eval_genomes, 100) # Increased generation count
    
    with open("best_bot_raycast.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("\nBest genome saved.")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run_training(config_path)
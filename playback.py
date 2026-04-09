import pymunk
import pygame
import neat
import pickle
import math
import os
import random

# --- CONSTANTS (Must match Training) ---
WIDTH, HEIGHT = 800, 800
NUM_RAYS = 8
SENSE_DIST = 350  # Matches the updated training distance
FPS = 60

def get_inputs(bot_body, bot_vel, hazards, space):
    # NEW: Boundary Distance Inputs (Normalized 0 to 1)
    # 0.0 means touching the wall, 1.0 means at the far opposite side
    dist_left = bot_body.position.x / WIDTH
    dist_right = (WIDTH - bot_body.position.x) / WIDTH
    dist_top = bot_body.position.y / HEIGHT
    dist_bottom = (HEIGHT - bot_body.position.y) / HEIGHT
    
    wall_data = [dist_left, dist_right, dist_top, dist_bottom]
    # 1. Raycasting (8 inputs)
    sensors = []
    for i in range(NUM_RAYS):
        angle = (math.pi * 2 / NUM_RAYS) * i
        ray_vec = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        start = bot_body.position
        end = start + ray_vec * SENSE_DIST
        info = space.segment_query_first(start, end, 1, pymunk.ShapeFilter())
        sensors.append(info.alpha if info else 1.0)
    
    # 2. Sort hazards and prepare relative data
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

    # 4. Cross Product: Are hazards pinning the bot?
    if len(rel_vectors) == 2:
        pincer_risk = abs(rel_vectors[0].cross(rel_vectors[1]))

    # 5. Normalized Bot Position
    bot_pos_data = [bot_body.position.x / WIDTH, bot_body.position.y / HEIGHT]

    # Total = 8 (rays) + 8 (hazards) + 2 (pos) + 2 (dots) + 1 (cross) = 21
    return sensors + h_data + bot_pos_data + dot_products + [pincer_risk]+wall_data
def run_best_bot(config_path, genome_path):
    current_bot_vel = pymunk.Vec2d(0, 0)

    if not os.path.exists(genome_path):
        print(f"Error: {genome_path} not found. Train the bot first!")
        return

    with open(genome_path, "rb") as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neural Network Best Bot - Playback (21 Inputs)")
    clock = pygame.time.Clock()
    space = pymunk.Space()
    
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = (WIDTH/2, HEIGHT/2)
    c_shape = pymunk.Circle(body, 12)
    space.add(body, c_shape)

    hazards = []
    spawn_timer = 0
    font = pygame.font.SysFont("Arial", 20)

    while True:
        dt = 1/FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # --- THINKING ---
        inputs = get_inputs(body, current_bot_vel, hazards, space)
        output = net.activate(inputs)

        target_vx = output[0] * 450 
        target_vy = output[1] * 450
        
        # --- ACTING ---
        move_speed = 500 * dt
        current_bot_vel = pymunk.Vec2d(target_vx, target_vy)
        body.position += current_bot_vel * dt

        # 4. BOUNDARY & CENTER LOGIC
        body.position = (max(20, min(WIDTH-20, body.position.x)), max(20, min(HEIGHT-20, body.position.y)))

        # --- BOUNDARY ENFORCEMENT (Matched to Training) ---
        body.position = (
            max(15, min(WIDTH - 15, body.position.x)),
            max(15, min(HEIGHT - 15, body.position.y))
        )

        # --- SPAWNING ---
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
            else: 
                h_body.position = (WIDTH + 50, random.randint(0, HEIGHT))
                h_body.velocity = (random.randint(-180, -60), random.randint(-120, 120))

            h_shape = pymunk.Poly.create_box(h_body, (40, 40))
            space.add(h_body, h_shape)
            hazards.append(h_shape)
        spawn_timer -= 1

        space.step(dt)
        screen.fill((20, 24, 35))
        
        # Draw and update hazards
        for h in hazards[:]:
            h.body.position += h.body.velocity * dt
            
            # Accurate Collision Check
            query = h.point_query(body.position)
            if query.distance < 12:
                print("Collision! Resetting simulation...")
                body.position = (WIDTH/2, HEIGHT/2)
                for item in hazards: space.remove(item.body, item)
                hazards.clear()
                break

            # Boundary Cleanup
            if h.body.position.x < -100 or h.body.position.x > WIDTH+100 or \
               h.body.position.y < -100 or h.body.position.y > HEIGHT+100:
                space.remove(h.body, h)
                hazards.remove(h)
                continue

            pts = [h.body.local_to_world(v) for v in h.get_vertices()]
            pygame.draw.polygon(screen, (220, 60, 60), pts)

        # Draw Vision Rays
        for i in range(NUM_RAYS):
            angle = (math.pi * 2 / NUM_RAYS) * i
            end = body.position + pymunk.Vec2d(math.cos(angle), math.sin(angle)) * SENSE_DIST
            pygame.draw.line(screen, (60, 60, 80), body.position, end, 1)
            
        pygame.draw.circle(screen, (255, 215, 0), (int(body.position.x), int(body.position.y)), 12)
        screen.blit(font.render(f"Best Bot Playback - Inputs: 21", True, (255, 255, 255)), (20, 20))
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_p = os.path.join(local_dir, 'config-feedforward.txt')
    genome_p = os.path.join(local_dir, 'best_bot_raycast.pkl')
    run_best_bot(config_p, genome_p)
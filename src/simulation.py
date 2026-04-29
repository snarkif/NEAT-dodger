import pymunk
import random
import math

FPS = 60
WIDTH, HEIGHT = 800, 800
NUM_RAYS = 8
SENSE_DIST = 350


def get_inputs(bot_body, bot_vel, hazards, space):
    dist_left = bot_body.position.x / WIDTH
    dist_right = (WIDTH - bot_body.position.x) / WIDTH
    dist_top = bot_body.position.y / HEIGHT
    dist_bottom = (HEIGHT - bot_body.position.y) / HEIGHT

    wall_data = [dist_left, dist_right, dist_top, dist_bottom]

    sensors = []
    for i in range(NUM_RAYS):
        angle = (math.pi * 2 / NUM_RAYS) * i
        ray_vec = pymunk.Vec2d(math.cos(angle), math.sin(angle))
        start = bot_body.position
        end = start + ray_vec * SENSE_DIST
        info = space.segment_query_first(start, end, 1, pymunk.ShapeFilter())
        sensors.append(info.alpha if info else 1.0)

    hazards.sort(key=lambda h: (h.body.position - bot_body.position).length)

    h_data = [0.0] * 8
    dot_products = [0.0, 0.0]
    pincer_risk = 0.0
    rel_vectors = []

    for i in range(min(len(hazards), 2)):
        h = hazards[i]
        rel = (h.body.position - bot_body.position) / SENSE_DIST
        rel_vectors.append(rel)

        idx = i * 4
        h_data[idx] = max(-1.0, min(1.0, rel.x))
        h_data[idx + 1] = max(-1.0, min(1.0, rel.y))
        h_data[idx + 2] = h.body.velocity.x / 200
        h_data[idx + 3] = h.body.velocity.y / 200

        if bot_vel.length > 0:
            dot_products[i] = bot_vel.normalized().dot(rel.normalized())

    if len(rel_vectors) == 2:
        pincer_risk = abs(rel_vectors[0].cross(rel_vectors[1]))

    bot_pos_data = [bot_body.position.x / WIDTH, bot_body.position.y / HEIGHT]

    return sensors + h_data + bot_pos_data + dot_products + [pincer_risk] + wall_data


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        import neat

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        s = pymunk.Space()

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
        b.position = (WIDTH / 2, HEIGHT / 2)
        c = pymunk.Circle(b, 12)
        s.add(b, c)

        hazards = []
        running = True
        spawn_timer = 0
        dt = 1 / FPS

        current_vel = pymunk.Vec2d(0, 0)
        prev_dist_to_hazards = {}

        for frame_count in range(5000):
            if not running:
                break

            inputs = get_inputs(b, current_vel, hazards, s)
            output = net.activate(inputs)

            target_vx = output[0] * 450
            target_vy = output[1] * 450
            new_vel = pymunk.Vec2d(target_vx, target_vy)

            accel = (new_vel - current_vel).length
            genome.fitness -= (accel / 500) * 0.1

            current_vel = new_vel
            b.position += current_vel * dt

            if (
                b.position.x < 40 or b.position.x > WIDTH - 40 or
                b.position.y < 40 or b.position.y > HEIGHT - 40
            ):
                genome.fitness -= 1.2

            b.position = (
                max(20, min(WIDTH - 20, b.position.x)),
                max(20, min(HEIGHT - 20, b.position.y))
            )

            genome.fitness += 1

            if current_vel.length > 50:
                genome.fitness += 0.2

            if hazards:
                total_proximity_pressure = 0
                for i, h in enumerate(hazards[:3]):
                    dist = (h.body.position - b.position).length

                    if dist < 120:
                        proximity_loss = (120 - dist) / 100
                        total_proximity_pressure += proximity_loss
                        genome.fitness -= total_proximity_pressure * 2.0

                    if i in prev_dist_to_hazards and dist < 200:
                        if dist > prev_dist_to_hazards[i] + 0.2:
                            genome.fitness += 2

                    prev_dist_to_hazards[i] = dist

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
                s.add(h_body, h_shape)
                hazards.append(h_shape)

            spawn_timer -= 1
            s.step(dt)

            for shape in hazards[:]:
                shape.body.position += shape.body.velocity * dt

                if shape.point_query(b.position).distance < 12:
                    genome.fitness -= 400
                    running = False

                if (
                    shape.body.position.x < -100 or shape.body.position.x > WIDTH + 100 or
                    shape.body.position.y < -100 or shape.body.position.y > HEIGHT + 100
                ):
                    s.remove(shape.body, shape)
                    hazards.remove(shape)
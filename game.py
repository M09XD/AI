from pickletools import optimize
import pygame
import random
import math
from collections import defaultdict
import copy

# --- Core Classes ---
class Move:
    def __init__(self, name, move_type, power, accuracy):
        self.name = name
        self.type = move_type
        self.power = power
        self.accuracy = accuracy

class Pokemon:
    def __init__(self, name, p_type, hp, attack, defense, moves):
        self.name = name
        self.type = p_type
        self.max_hp = hp
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.moves = moves

    def take_damage(self, damage):
        self.hp = max(0, self.hp - damage)

    def is_fainted(self):
        if self.hp <= 0:
            return True
        else:
            return False

    def heal(self):
        self.hp = self.max_hp

# Type effectiveness chart
type_chart = {
    'Fire': {'Grass': 2.0, 'Water': 0.5, 'Fire': 0.5},
    'Water': {'Fire': 2.0, 'Grass': 0.5, 'Water': 0.5},
    'Grass': {'Water': 2.0, 'Fire': 0.5, 'Grass': 0.5},
}

def type_effectiveness(attacking_type, defending_type):
    return type_chart.get(attacking_type, {}).get(defending_type, 1.0)

# Function to calculate damage
def calculate_damage(attacker, defender, move):
    if random.random() > move.accuracy:
        return 0
    effectiveness = type_effectiveness(move.type, defender.type)
    base_damage = ((2 * attacker.attack / max(1, defender.defense)) * move.power / 50) + 2
    return int(base_damage * effectiveness * random.uniform(0.85, 1.0))
    



# --- Minimax with Alpha-Beta Pruning ---

class BattleState:
    def __init__(self, player1, player2, turn=1):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.turn = turn  # 1 for player1, -1 for player2

    def is_terminal(self):
        return self.player1.is_fainted() or self.player2.is_fainted()

    def get_winner(self):
        if self.player1.is_fainted() and not self.player2.is_fainted():
            return 'Player 2'
        elif self.player2.is_fainted() and not self.player1.is_fainted():
            return 'Player 1'
        return 'Draw'

def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return evaluate_state(state), None

    best_move = None
    player = state.player1 if maximizing_player else state.player2
    opponent = state.player2 if maximizing_player else state.player1

    if maximizing_player:
        max_eval = float('-inf')
        for move in player.moves:
            new_state = simulate_move(state, move, maximizing_player)
            eval, _ = minimax(new_state, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in player.moves:
            new_state = simulate_move(state, move, maximizing_player)
            eval, _ = minimax(new_state, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move
    
def simulate_move(state, move, is_player1_turn):
    new_state = copy.deepcopy(state)
    attacker = new_state.player1 if is_player1_turn else new_state.player2
    defender = new_state.player2 if is_player1_turn else new_state.player1
    damage = calculate_damage(attacker, defender, move)
    defender.take_damage(damage)
    return new_state

def evaluate_state(state):
    return state.player1.hp - state.player2.hp



# --- Naive Bayes ---

class NaiveBayesPredictor:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)

    def train(self, sequences):
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                prev_move, next_move = sequence[i], sequence[i + 1]
                self.transition_counts[prev_move][next_move] += 1
                self.total_counts[prev_move] += 1

    # Inside NaiveBayesPredictor
    def predict(self, current_move):
        if current_move not in self.transition_counts:
           return None
        next_moves = self.transition_counts[current_move]
        if self.total_counts[current_move] == 0:
           return None
        return max(next_moves, key=lambda move: next_moves[move] / self.total_counts[current_move])


# --- Simulated Annealing ---
def simulated_annealing(initial_moves, state, max_iter=1000, temp=100.0, cooling_rate=0.95):
    current = initial_moves[:]
    best = current[:]
    best_score = -float('inf')

    for i in range(max_iter):
        new = current[:]
        idx1, idx2 = random.sample(range(len(new)), 2)
        new[idx1], new[idx2] = new[idx2], new[idx1]

        score = 0
        new_state = copy.deepcopy(state)

        for move in new:
            if new_state.is_terminal():
                break
            new_state = simulate_move(new_state, move, True)

        score = evaluate_state(new_state)

        if score > best_score or random.random() < math.exp((score - best_score) / temp):
            current = new[:]
            best_score = score
            best = new[:]

        temp *= cooling_rate
    return best


# --- CSP (Constraint Satisfaction Problem) ---

def is_valid_move_sequence(sequence):
    for i in range(len(sequence) - 1):
        if sequence[i].name == sequence[i + 1].name:
            return False
    return True

# Sample moves and Pokémon
tackle = Move("Tackle", "Normal", 40, 1.0)
ember = Move("Ember", "Fire", 40, 1.0)
water_gun = Move("Water Gun", "Water", 40, 1.0)
vine_whip = Move("Vine Whip", "Grass", 40, 1.0)

charmander = Pokemon("Charmander", "Fire", 100, 52, 43, [tackle, ember])
squirtle = Pokemon("Squirtle", "Water", 100, 48, 65, [tackle, water_gun])
bulbasaur = Pokemon("Bulbasaur", "Grass", 100, 49, 49, [tackle, vine_whip])


player_pokemon = random.choice([charmander, squirtle, bulbasaur])
opponent_pokemon = random.choice([charmander, squirtle, bulbasaur])


while player_pokemon == opponent_pokemon:
    opponent_pokemon = random.choice([charmander, squirtle, bulbasaur])

from itertools import product

def generate_valid_sequences(pokemon1, pokemon2, length=3):
    all_moves = [move.name for move in pokemon1.moves + pokemon2.moves]
    sequences = []
    for combo in product(all_moves, repeat=length):
        if all(combo[i] != combo[i+1] for i in range(len(combo)-1)):  # no consecutive repeats
            sequences.append(list(combo))
    return sequences


nb = NaiveBayesPredictor()
training_sequences = generate_valid_sequences(player_pokemon, opponent_pokemon, length=3)
nb.train(training_sequences)
##
print(f"Training {player_pokemon.name} vs {opponent_pokemon.name} with {len(training_sequences)} sequences")


current_move_name = "Tackle"

move_index = 0  # For iterating through optimized move order

# --- Pygame Setup ---

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pokémon AI Battle Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
bg_color = (255, 255, 255)

# --- Rectangles ---
player_rect = pygame.Rect(100, 300, 120, 120)
ai_rect = pygame.Rect(580, 100, 120, 120)

def draw_hp_bar(pokemon, x, y):
    bar_width = 100
    hp_ratio = pokemon.hp / pokemon.max_hp
    pygame.draw.rect(screen, (0, 0, 0), (x, y, bar_width, 10), 1)
    pygame.draw.rect(screen, (0, 255, 0), (x, y, int(bar_width * hp_ratio), 10))

def draw_dialog(text, moves=None):
    box_x, box_y, box_w, box_h = 50, 450, 700, 130
    pygame.draw.rect(screen, (230, 230, 230), (box_x, box_y, box_w, box_h))  # Background
    pygame.draw.rect(screen, (0, 0, 0), (box_x, box_y, box_w, box_h), 2)     # Border

    padding = 10
    if moves:
        # Moves list still goes to the top-left
        y_offset = box_y + padding
        for i, move in enumerate(moves):
            rendered = font.render(f"{i+1}: {move.name}", True, (0, 0, 0))
            screen.blit(rendered, (box_x + padding, y_offset))
            y_offset += 20

    # Dialog text in bottom-right
    lines = text.split("\n")
    line_spacing = 25
    total_text_height = len(lines) * line_spacing
    y_start = box_y + box_h - total_text_height - padding
    for i, line in enumerate(lines):
        rendered = font.render(line, True, (0, 0, 0))
        text_width = rendered.get_width()
        x_pos = box_x + box_w - text_width - padding
        y_pos = y_start + i * line_spacing
        screen.blit(rendered, (x_pos, y_pos))


def draw_battle_screen(dialog_text=""):
    screen.fill(bg_color)
    pygame.draw.rect(screen, (255, 100, 100), player_rect)
    pygame.draw.rect(screen, (100, 100, 255), ai_rect)
    pygame.draw.rect(screen, (0, 0, 0), player_rect, 2)
    pygame.draw.rect(screen, (0, 0, 0), ai_rect, 2)

    player_name = font.render(player_pokemon.name, True, (0, 0, 0))
    ai_name = font.render(opponent_pokemon.name, True, (0, 0, 0))
    screen.blit(player_name, (player_rect.x, player_rect.y - 35))
    screen.blit(ai_name, (ai_rect.x, ai_rect.y - 35))

    draw_hp_bar(player_pokemon, player_rect.x, player_rect.y - 20)
    draw_hp_bar(opponent_pokemon, ai_rect.x, ai_rect.y - 20)

    draw_dialog(dialog_text, player_pokemon.moves)



def show_move_dialog(attacker, defender, move, damage):
    effectiveness = type_effectiveness(move.type, defender.type)
    
    # Base move message
    dialog = f"{attacker.name} used {move.name}!"
    draw_battle_screen(dialog)
    pygame.display.flip()
    pygame.time.wait(1500)

    # Effectiveness message
    if damage == 0:
        dialog = "But it missed!"
    elif effectiveness > 1.0:
        dialog = "It's super effective!"
    elif effectiveness < 1.0:
        dialog = "It's not very effective..."
    else:
        dialog = ""

    if dialog:
        draw_battle_screen(dialog)
        pygame.display.flip()
        pygame.time.wait(1500)


def draw_menu():
    screen.fill(bg_color)
    title = font.render("Choose AI Strategy:", True, (0, 0, 0))
    option1 = font.render("1: MinMax", True, (0, 0, 0))
    option2 = font.render("2: Simulated Annealing", True, (0, 0, 0))
    option3 = font.render("3: Naive Bayes", True, (0, 0, 0))
    option4 = font.render("4: Random", True, (0, 0, 0))
    screen.blit(title, (WIDTH//2 - 100, HEIGHT//2 - 60))
    screen.blit(option1, (WIDTH//2 - 100, HEIGHT//2 - 30))
    screen.blit(option2, (WIDTH//2 - 100, HEIGHT//2))
    screen.blit(option3, (WIDTH//2 - 100, HEIGHT//2 + 30))
    screen.blit(option4, (WIDTH//2 - 100, HEIGHT//2 + 60))
    pygame.display.flip()


from itertools import product

def generate_valid_sequences(pokemon1, pokemon2, length=3):
    all_moves = [move.name for move in pokemon1.moves + pokemon2.moves]
    sequences = []
    for combo in product(all_moves, repeat=length):
        if all(combo[i] != combo[i+1] for i in range(len(combo)-1)):  # avoid same move twice in a row
            sequences.append(list(combo))
    return sequences


# --- Main Game Loop ---
running = True
turn = 'player'
selected_move = None

# --- AI Selection Menu ---
ai_strategy = None
while ai_strategy is None:
    draw_menu()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                ai_strategy = '1'
            elif event.key == pygame.K_2:
                ai_strategy = '2'
            elif event.key == pygame.K_3:
                ai_strategy = '3'
            elif event.key == pygame.K_4:
                ai_strategy = '4'
            else:
                ai_strategy = None


while running:
    clock.tick(30)
    draw_battle_screen()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN and turn == 'player':
            if event.key in (pygame.K_1, pygame.K_2):
                selected_index = event.key - pygame.K_1
                if 0 <= selected_index < len(player_pokemon.moves):
                    selected_move = player_pokemon.moves[selected_index]
                    damage = calculate_damage(player_pokemon, opponent_pokemon, selected_move)
                    opponent_pokemon.take_damage(damage)

                    # Show dialog with delay
                    show_move_dialog(player_pokemon, opponent_pokemon, selected_move, damage)

                    turn = 'ai'



    if turn == 'ai' and not opponent_pokemon.is_fainted():
        pygame.time.delay(500)

        if ai_strategy == '1':
            # Use Minimax to pick the best move for opponent
            state = BattleState(player_pokemon, opponent_pokemon, turn=-1)
            _, ai_move = minimax(state, depth=2, alpha=-float('inf'), beta=float('inf'), maximizing_player=False)
            if ai_move is None:
                ai_move = random.choice(opponent_pokemon.moves)
        

        elif ai_strategy == '2':
            state = BattleState(player_pokemon, opponent_pokemon, turn=-1)
            optimized_moves = simulated_annealing(opponent_pokemon.moves, state)
            ai_move = optimized_moves[0] if optimized_moves else random.choice(opponent_pokemon.moves)

        
        
        elif ai_strategy == '3':
            predicted_move_name = nb.predict(current_move_name)
            ai_move = next((move for move in opponent_pokemon.moves if move.name == predicted_move_name), random.choice(opponent_pokemon.moves))
            current_move_name = ai_move.name
        
        
        else:
            ai_move = random.choice(opponent_pokemon.moves)
        
        
        damage = calculate_damage(opponent_pokemon, player_pokemon, ai_move)
        player_pokemon.take_damage(damage)

        # Show dialog with delay
        show_move_dialog(opponent_pokemon, player_pokemon, ai_move, damage)

        turn = 'player'




    if player_pokemon.is_fainted() or opponent_pokemon.is_fainted():
        winner_text = "Draw!"
        if player_pokemon.is_fainted():
            winner_text = f"{opponent_pokemon.name} wins!"
        elif opponent_pokemon.is_fainted():
            winner_text = f"{player_pokemon.name} wins!"



        end_text = font.render(winner_text, True, (0, 0, 0))
        screen.blit(end_text, (WIDTH//2 - 50, HEIGHT//2))
        pygame.display.flip()
        pygame.time.wait(3000)
        running = False

pygame.quit()

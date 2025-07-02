import pygame
import random
import math
import os
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
        return self.hp <= 0

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

def calculate_damage(attacker, defender, move):
    if random.random() > move.accuracy:
        return 0, "missed"
    effectiveness = type_effectiveness(move.type, defender.type)
    base_damage = ((2 * attacker.attack / max(1, defender.defense)) * move.power / 50) + 2
    damage = int(base_damage * effectiveness * random.uniform(0.85, 1.0))
    if effectiveness > 1:
        return damage, "super effective"
    elif effectiveness < 1:
        return damage, "not very effective"
    return damage, "normal"

# --- AI Algorithms ---
class BattleState:
    def __init__(self, player1, player2, turn=1):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.turn = turn

    def is_terminal(self):
        return self.player1.is_fainted() or self.player2.is_fainted()

def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return state.player1.hp - state.player2.hp, None

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
    damage, _ = calculate_damage(attacker, defender, move)
    defender.take_damage(damage)
    return new_state

class NaiveBayesPredictor:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)

    def train(self, sequences):
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                prev, nxt = sequence[i], sequence[i+1]
                self.transition_counts[prev][nxt] += 1
                self.total_counts[prev] += 1

    def predict(self, current):
        if current not in self.transition_counts or self.total_counts[current] == 0:
            return None
        next_moves = self.transition_counts[current]
        return max(next_moves, key=lambda m: next_moves[m] / self.total_counts[current])

    def update_from_file(self, filename):
        if not os.path.exists(filename): return
        with open(filename, 'r') as f:
            data = [line.strip() for line in f.readlines() if line.strip()]
            self.train([data])

    def save_move(self, filename, move):
        with open(filename, 'a') as f:
            f.write(f"{move}\n")

# --- Setup Game Data ---
move_log_file = "move_data.txt"
nb = NaiveBayesPredictor()
nb.update_from_file(move_log_file)

moves = [Move("Tackle", "Normal", 40, 1.0), Move("Ember", "Fire", 40, 1.0), Move("Water Gun", "Water", 40, 1.0), Move("Vine Whip", "Grass", 45, 1.0)]
charmander = Pokemon("Charmander", "Fire", 100, 52, 43, [moves[1], moves[0]])
squirtle = Pokemon("Squirtle", "Water", 100, 48, 65, [moves[2], moves[0]])
bulbasaur = Pokemon("Bulbasaur", "Grass", 100, 49, 49, [moves[3], moves[0]])

player_pokemon = random.choice([charmander, squirtle, bulbasaur])
opponent_pokemon = random.choice([charmander, squirtle, bulbasaur])
while player_pokemon.name == opponent_pokemon.name:
    opponent_pokemon = random.choice([charmander, squirtle, bulbasaur])

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()
background = pygame.Surface(screen.get_size())
background.fill((255, 255, 255))

# --- GUI Helpers ---
def draw_hp_bar(pokemon, x, y):
    bar_width = 100
    hp_ratio = pokemon.hp / pokemon.max_hp
    pygame.draw.rect(screen, (0, 0, 0), (x, y, bar_width, 10), 1)
    pygame.draw.rect(screen, (0, 255, 0), (x, y, int(bar_width * hp_ratio), 10))

def draw_dialog(text):
    pygame.draw.rect(screen, (230, 230, 230), (50, 450, 700, 100))
    pygame.draw.rect(screen, (0, 0, 0), (50, 450, 700, 100), 2)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        rendered = font.render(line, True, (0, 0, 0))
        screen.blit(rendered, (60, 460 + i*25))

# --- Main Game Loop ---
running = True
dialog = "A wild battle begins!"
turn = 'player'
current_move_name = "Tackle"

while running:
    screen.blit(background, (0, 0))

    # Draw Pokémon
    screen.blit(font.render(player_pokemon.name, True, (0, 0, 0)), (100, 300))
    draw_hp_bar(player_pokemon, 100, 330)
    screen.blit(font.render(opponent_pokemon.name, True, (0, 0, 0)), (600, 100))
    draw_hp_bar(opponent_pokemon, 600, 130)
    draw_dialog(dialog)

    for i, move in enumerate(player_pokemon.moves):
        move_txt = font.render(f"{i+1}: {move.name}", True, (0, 0, 0))
        screen.blit(move_txt, (50, 380 + i * 25))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and turn == 'player':
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key in (pygame.K_1, pygame.K_2):
                idx = event.key - pygame.K_1
                if idx < len(player_pokemon.moves):
                    move = player_pokemon.moves[idx]
                    damage, msg = calculate_damage(player_pokemon, opponent_pokemon, move)
                    opponent_pokemon.take_damage(damage)
                    dialog = f"{player_pokemon.name} used {move.name}!\n{msg.title()}!\nIt dealt {damage} damage."
                    nb.save_move(move_log_file, move.name)
                    turn = 'ai'

    if turn == 'ai' and not opponent_pokemon.is_fainted():
        pygame.time.delay(500)
        state = BattleState(opponent_pokemon, player_pokemon, -1)
        _, ai_move = minimax(state, 2, -math.inf, math.inf, True)
        if ai_move is None:
            ai_move = random.choice(opponent_pokemon.moves)
        damage, msg = calculate_damage(opponent_pokemon, player_pokemon, ai_move)
        player_pokemon.take_damage(damage)
        dialog = f"{opponent_pokemon.name} used {ai_move.name}!\n{msg.title()}!\nIt dealt {damage} damage."
        current_move_name = ai_move.name
        turn = 'player'

    if player_pokemon.is_fainted() or opponent_pokemon.is_fainted():
        if player_pokemon.is_fainted() and opponent_pokemon.is_fainted():
            dialog = "Both Pokémon fainted! It's a draw."
        elif player_pokemon.is_fainted():
            dialog = f"{player_pokemon.name} fainted! You are defeated."
        else:
            dialog = f"{opponent_pokemon.name} fainted! {player_pokemon.name} won!"
        draw_dialog(dialog)
        pygame.display.flip()
        pygame.time.wait(3000)
        running = False

pygame.quit()

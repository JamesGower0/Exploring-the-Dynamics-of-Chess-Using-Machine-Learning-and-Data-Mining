import chess
import chess.pgn
import chess.engine
import csv
from multiprocessing import Pool, cpu_count
import time
from functools import lru_cache

STOCKFISH_PATH = "..\stockfish\stockfish-windows-x86-64-avx2.exe"
THREADS = 8  # Match your CPU core count
HASH_SIZE = 2048  # 2GB hash - adjust based on your RAM (32GB total)
EVAL_DEPTH = 10  # Keep your original depth

def init_engine():
    """Initialize a Stockfish engine with optimal settings"""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({
        "Threads": THREADS,
        "Hash": HASH_SIZE,
        "Skill Level": 20,
        "UCI_LimitStrength": False
    })
    return engine

@lru_cache(maxsize=10000)
def cached_evaluate(board, limit_depth=EVAL_DEPTH):
    """Cached evaluation function to avoid recomputing the same positions"""
    engine = init_engine()
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=limit_depth))
        score = info["score"].relative
        
        if score.is_mate():
            mate_in = score.mate()
            return 100 if mate_in > 0 else -100
        
        evaluation = score.score(mate_score=10000) / 100.0
        if board.turn == chess.BLACK:
            evaluation *= -1
            
        return evaluation
    finally:
        engine.quit()
        
def detect_battery(board):
    """Optimized battery detection using bitboards"""
    battery_detected = False
    battery_colour = None
    
    for color in [chess.WHITE, chess.BLACK]:
        # Get all aligned pieces (queens, rooks, bishops)
        queens = board.queens(color)
        rooks = board.rooks(color)
        bishops = board.bishops(color)
        
        aligned_pieces = queens | rooks | bishops
        
        for piece_square in chess.scan_reversed(aligned_pieces):
            piece_type = board.piece_type_at(piece_square)
            rank = chess.square_rank(piece_square)
            
            # Skip starting ranks
            if (color == chess.WHITE and rank == 7) or (color == chess.BLACK and rank == 0):
                continue
                
            # Get attack mask
            attacks = board.attacks(piece_square)
            
            # Look for aligned pieces in attack path
            for target_square in chess.scan_reversed(attacks & aligned_pieces):
                if target_square == piece_square:
                    continue
                    
                # Check if they're aligned in a straight line
                if chess.square_distance(piece_square, target_square) > 1:
                    direction = chess.square_direction(piece_square, target_square)
                    between = chess.ray(piece_square, target_square)
                    
                    # Check if path is clear between pieces
                    if not any(board.piece_at(sq) for sq in between):
                        # Check if pointing to enemy piece
                        beyond = chess.ray(target_square, target_square + (target_square - piece_square))
                        for attack_square in beyond:
                            attack_piece = board.piece_at(attack_square)
                            if attack_piece and attack_piece.color != color:
                                battery_detected = True
                                battery_colour = color
                                return battery_detected, battery_colour
                                
    return battery_detected, battery_colour

def detect_fianchetto(board, move):
    """Detects if a fianchetto is used in the given move."""
    fianchetto_squares = {
        chess.BLACK: [chess.B2, chess.G2],
        chess.WHITE: [chess.B7, chess.G7]
    }
    
    if (move.to_square in fianchetto_squares[board.turn] and 
        board.piece_at(move.to_square) and 
        board.piece_at(move.to_square).piece_type == chess.BISHOP):
        return True, board.turn
    return False, None

def detect_bishop_pair(board):
    """Optimized bishop pair detection"""
    white_bishops = bin(board.pieces(chess.BISHOP, chess.WHITE)).count('1')
    black_bishops = bin(board.pieces(chess.BISHOP, chess.BLACK)).count('1')
    
    if white_bishops >= 2 and black_bishops < 2:
        return True, chess.WHITE
    elif black_bishops >= 2 and white_bishops < 2:
        return True, chess.BLACK
    return False, None

def process_game(game):
    """Process a single game with pattern detection and evaluation"""
    engine = init_engine()
    board = game.board()
    
    # Detection flags and tracking variables
    detections = {
        'battery': {'detected': False, 'colour': None, 'before': None, 'after': None, 'counter': 0},
        'fianchetto': {'detected': False, 'colour': None, 'before': None, 'after': None, 'counter': 0},
        'bishop_pair': {'detected': False, 'colour': None, 'before': None, 'after': None, 'counter': 0}
    }
    
    # Bishop pair tracking
    bishop_pair_counters = {chess.WHITE: 0, chess.BLACK: 0}
    
    for move in game.mainline_moves():
        # Make copies of board state before move
        pre_move_boards = {k: board.copy() for k in detections if not detections[k]['detected']}
        
        board.push(move)
        
        # Update counters
        for key in detections:
            if not detections[key]['detected']:
                detections[key]['counter'] += 1
        
        # Battery detection
        if not detections['battery']['detected']:
            battery_detected, battery_colour = detect_battery(board)
            if battery_detected:
                detections['battery'].update({
                    'detected': True,
                    'colour': battery_colour,
                    'before': cached_evaluate(pre_move_boards['battery']),
                    'counter': 0
                })
        
        # Fianchetto detection
        if not detections['fianchetto']['detected']:
            fianchetto_detected, fianchetto_colour = detect_fianchetto(pre_move_boards['fianchetto'], move)
            if fianchetto_detected:
                detections['fianchetto'].update({
                    'detected': True,
                    'colour': fianchetto_colour,
                    'before': cached_evaluate(pre_move_boards['fianchetto']),
                    'counter': 0
                })
        
        # Bishop pair detection
        if not detections['bishop_pair']['detected']:
            bishop_pair_detected, bishop_pair_colour = detect_bishop_pair(board)
            if bishop_pair_detected:
                if bishop_pair_colour == chess.WHITE:
                    bishop_pair_counters[chess.WHITE] += 1
                    bishop_pair_counters[chess.BLACK] = 0
                else:
                    bishop_pair_counters[chess.BLACK] += 1
                    bishop_pair_counters[chess.WHITE] = 0
                
                if bishop_pair_counters[bishop_pair_colour] >= 5:
                    detections['bishop_pair'].update({
                        'detected': True,
                        'colour': bishop_pair_colour,
                        'before': cached_evaluate(pre_move_boards['bishop_pair']),
                        'counter': 0
                    })
            else:
                bishop_pair_counters[chess.WHITE] = 0
                bishop_pair_counters[chess.BLACK] = 0
        
        # Post-detection evaluations
        for key in detections:
            if detections[key]['detected'] and detections[key]['after'] is None:
                if (key == 'fianchetto' and detections[key]['counter'] >= 10) or \
                   (key != 'fianchetto' and detections[key]['counter'] >= 5) or \
                   board.is_game_over():
                    detections[key]['after'] = cached_evaluate(board)
    
    # Final evaluations if not done during the game
    for key in detections:
        if detections[key]['detected'] and detections[key]['after'] is None:
            detections[key]['after'] = cached_evaluate(board)
    
    engine.quit()
    
    # Prepare results
    return [
        game.headers.get("Event", "Unknown"),
        game.headers.get("WhiteElo", "Unknown"),
        game.headers.get("BlackElo", "Unknown"),
        game.headers.get("Result", "Unknown"),
        int(detections['battery']['detected']),
        1 if detections['battery']['colour'] == chess.WHITE else 0 if detections['battery']['colour'] == chess.BLACK else None,
        detections['battery']['before'],
        detections['battery']['after'],
        int(detections['fianchetto']['detected']),
        0 if detections['fianchetto']['colour'] == chess.WHITE else 1 if detections['fianchetto']['colour'] == chess.BLACK else None,
        detections['fianchetto']['before'],
        detections['fianchetto']['after'],
        int(detections['bishop_pair']['detected']),
        1 if detections['bishop_pair']['colour'] == chess.WHITE else 0 if detections['bishop_pair']['colour'] == chess.BLACK else None,
        detections['bishop_pair']['before'],
        detections['bishop_pair']['after']
    ]

def process_pgn_parallel(pgn_path, output_csv, num_workers=None):
    """Process PGN file using multiple workers"""
    if num_workers is None:
        num_workers = cpu_count()
    
    with open(pgn_path) as pgn, open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Event", "WhiteElo", "BlackElo", "Result", 
            "BatteryUsed", "Batterycolour", "BatteryEvalBefore", "BatteryEvalAfter",
            "FianchettoUsed", "Fianchettocolour", "FianchettoEvalBefore", "FianchettoEvalAfter",
            "BishopPairUsed", "BishopPaircolour", "BishopPairEvalBefore", "BishopPairEvalAfter"
        ])
        
        # Create worker pool
        with Pool(num_workers) as pool:
            # Process games in batches
            batch_size = num_workers * 2
            games_processed = 0
            start_time = time.time()
            
            while True:
                # Read batch of games
                game_batch = []
                for _ in range(batch_size):
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    game_batch.append(game)
                
                if not game_batch:
                    break
                
                # Process batch in parallel
                results = pool.map(process_game, game_batch)
                
                # Write results
                for result in results:
                    writer.writerow(result)
                    csvfile.flush()  # Ensure data is written regularly
                
                # Progress reporting
                games_processed += len(game_batch)
                elapsed = time.time() - start_time
                games_per_sec = games_processed / elapsed
                print(f"\rProcessed {games_processed} games ({games_per_sec:.2f} games/sec)", end="", flush=True)
            
            print()  # New line after progress reporting

if __name__ == "__main__":
    process_pgn_parallel("SAN.pgn", "output2.csv")
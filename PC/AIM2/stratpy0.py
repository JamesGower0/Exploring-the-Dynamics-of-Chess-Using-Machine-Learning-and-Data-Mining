import chess
import chess.pgn
import chess.engine
import csv
from pathlib import Path

STOCKFISH_PATH = "..\stockfish\stockfish-windows-x86-64-avx2.exe"  # Updated Stockfish path

def detect_battery(board):
    """Detects if a valid battery (aligned pieces attacking an enemy piece) exists and returns the colour of the player using it."""
    battery_detected = False
    battery_colour = None  # Will store the colour of the player using the battery

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        # Define piece types for valid batteries
        diagonal_pieces = {chess.BISHOP, chess.QUEEN}
        straight_pieces = {chess.ROOK, chess.QUEEN}

        # Skip starting ranks
        rank = chess.square_rank(square)
        if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
            continue

        # Check along ranks (horizontal) for rooks and queens
        if piece.piece_type in straight_pieces:
            # Check to the right
            for offset in range(1, 8 - chess.square_file(square)):
                target_square = square + offset
                target_piece = board.piece_at(target_square)
                if target_piece:
                    if target_piece.color == piece.color and target_piece.piece_type in straight_pieces:
                        # Check if the battery is pointing towards an enemy piece
                        for attack_offset in range(1, 8 - chess.square_file(target_square)):
                            attack_square = target_square + attack_offset
                            attack_piece = board.piece_at(attack_square)
                            if attack_piece:
                                if attack_piece.color != piece.color:
                                    # Ensure no blocking pieces between the battery and the enemy piece
                                    blocking = False
                                    for block_offset in range(1, attack_offset):
                                        block_square = target_square + block_offset
                                        if board.piece_at(block_square):
                                            blocking = True
                                            break
                                    if not blocking:
                                        battery_detected = True
                                        battery_colour = piece.color  # Set the colour of the player using the battery
                                        break
                        if battery_detected:
                            break
                    break  # Stop if any piece is blocking

            # Check to the left
            if not battery_detected:
                for offset in range(1, chess.square_file(square) + 1):
                    target_square = square - offset
                    target_piece = board.piece_at(target_square)
                    if target_piece:
                        if target_piece.color == piece.color and target_piece.piece_type in straight_pieces:
                            # Check if the battery is pointing towards an enemy piece
                            for attack_offset in range(1, chess.square_file(target_square) + 1):
                                attack_square = target_square - attack_offset
                                attack_piece = board.piece_at(attack_square)
                                if attack_piece:
                                    if attack_piece.color != piece.color:
                                        # Ensure no blocking pieces between the battery and the enemy piece
                                        blocking = False
                                        for block_offset in range(1, attack_offset):
                                            block_square = target_square - block_offset
                                            if board.piece_at(block_square):
                                                blocking = True
                                                break
                                        if not blocking:
                                            battery_detected = True
                                            battery_colour = piece.color  # Set the colour of the player using the battery
                                            break
                            if battery_detected:
                                break
                        break  # Stop if any piece is blocking

        # Check along files (vertical) for rooks and queens
        if not battery_detected and piece.piece_type in straight_pieces:
            # Check downwards
            for offset in range(1, 8 - chess.square_rank(square)):
                target_square = square + 8 * offset
                target_piece = board.piece_at(target_square)
                if target_piece:
                    if target_piece.color == piece.color and target_piece.piece_type in straight_pieces:
                        # Check if the battery is pointing towards an enemy piece
                        for attack_offset in range(1, 8 - chess.square_rank(target_square)):
                            attack_square = target_square + 8 * attack_offset
                            attack_piece = board.piece_at(attack_square)
                            if attack_piece:
                                if attack_piece.color != piece.color:
                                    # Ensure no blocking pieces between the battery and the enemy piece
                                    blocking = False
                                    for block_offset in range(1, attack_offset):
                                        block_square = target_square + 8 * block_offset
                                        if board.piece_at(block_square):
                                            blocking = True
                                            break
                                    if not blocking:
                                        battery_detected = True
                                        battery_colour = piece.color  # Set the colour of the player using the battery
                                        break
                        if battery_detected:
                            break
                    break  # Stop if any piece is blocking

            # Check upwards
            if not battery_detected:
                for offset in range(1, chess.square_rank(square) + 1):
                    target_square = square - 8 * offset
                    target_piece = board.piece_at(target_square)
                    if target_piece:
                        if target_piece.color == piece.color and target_piece.piece_type in straight_pieces:
                            # Check if the battery is pointing towards an enemy piece
                            for attack_offset in range(1, chess.square_rank(target_square) + 1):
                                attack_square = target_square - 8 * attack_offset
                                attack_piece = board.piece_at(attack_square)
                                if attack_piece:
                                    if attack_piece.color != piece.color:
                                        # Ensure no blocking pieces between the battery and the enemy piece
                                        blocking = False
                                        for block_offset in range(1, attack_offset):
                                            block_square = target_square - 8 * block_offset
                                            if board.piece_at(block_square):
                                                blocking = True
                                                break
                                        if not blocking:
                                            battery_detected = True
                                            battery_colour = piece.color  # Set the colour of the player using the battery
                                            break
                            if battery_detected:
                                break
                        break  # Stop if any piece is blocking

        # Check along diagonals (positive slope) for bishops and queens
        if not battery_detected and piece.piece_type in diagonal_pieces:
            # Check downwards and to the right
            for offset in range(1, min(8 - chess.square_rank(square), 8 - chess.square_file(square))):
                target_square = square + 9 * offset
                target_piece = board.piece_at(target_square)
                if target_piece:
                    if target_piece.color == piece.color and target_piece.piece_type in diagonal_pieces:
                        # Check if the battery is pointing towards an enemy piece
                        for attack_offset in range(1, min(8 - chess.square_rank(target_square), 8 - chess.square_file(target_square))):
                            attack_square = target_square + 9 * attack_offset
                            attack_piece = board.piece_at(attack_square)
                            if attack_piece:
                                if attack_piece.color != piece.color:
                                    # Ensure no blocking pieces between the battery and the enemy piece
                                    blocking = False
                                    for block_offset in range(1, attack_offset):
                                        block_square = target_square + 9 * block_offset
                                        if board.piece_at(block_square):
                                            blocking = True
                                            break
                                    if not blocking:
                                        battery_detected = True
                                        battery_colour = piece.color  # Set the colour of the player using the battery
                                        break
                        if battery_detected:
                            break
                    break  # Stop if any piece is blocking

            # Check upwards and to the left
            if not battery_detected:
                for offset in range(1, min(chess.square_rank(square) + 1, chess.square_file(square) + 1)):
                    target_square = square - 9 * offset
                    target_piece = board.piece_at(target_square)
                    if target_piece:
                        if target_piece.color == piece.color and target_piece.piece_type in diagonal_pieces:
                            # Check if the battery is pointing towards an enemy piece
                            for attack_offset in range(1, min(chess.square_rank(target_square) + 1, chess.square_file(target_square) + 1)):
                                attack_square = target_square - 9 * attack_offset
                                attack_piece = board.piece_at(attack_square)
                                if attack_piece:
                                    if attack_piece.color != piece.color:
                                        # Ensure no blocking pieces between the battery and the enemy piece
                                        blocking = False
                                        for block_offset in range(1, attack_offset):
                                            block_square = target_square - 9 * block_offset
                                            if board.piece_at(block_square):
                                                blocking = True
                                                break
                                        if not blocking:
                                            battery_detected = True
                                            battery_colour = piece.color  # Set the colour of the player using the battery
                                            break
                            if battery_detected:
                                break
                        break  # Stop if any piece is blocking

        # Check along diagonals (negative slope) for bishops and queens
        if not battery_detected and piece.piece_type in diagonal_pieces:
            # Check downwards and to the left
            for offset in range(1, min(8 - chess.square_rank(square), chess.square_file(square) + 1)):
                target_square = square + 7 * offset
                target_piece = board.piece_at(target_square)
                if target_piece:
                    if target_piece.color == piece.color and target_piece.piece_type in diagonal_pieces:
                        # Check if the battery is pointing towards an enemy piece
                        for attack_offset in range(1, min(8 - chess.square_rank(target_square), chess.square_file(target_square) + 1)):
                            attack_square = target_square + 7 * attack_offset
                            attack_piece = board.piece_at(attack_square)
                            if attack_piece:
                                if attack_piece.color != piece.color:
                                    # Ensure no blocking pieces between the battery and the enemy piece
                                    blocking = False
                                    for block_offset in range(1, attack_offset):
                                        block_square = target_square + 7 * block_offset
                                        if board.piece_at(block_square):
                                            blocking = True
                                            break
                                    if not blocking:
                                        battery_detected = True
                                        battery_colour = piece.color  # Set the colour of the player using the battery
                                        break
                        if battery_detected:
                            break
                    break  # Stop if any piece is blocking

            # Check upwards and to the right
            if not battery_detected:
                for offset in range(1, min(chess.square_rank(square) + 1, 8 - chess.square_file(square))):
                    target_square = square - 7 * offset
                    target_piece = board.piece_at(target_square)
                    if target_piece:
                        if target_piece.color == piece.color and target_piece.piece_type in diagonal_pieces:
                            # Check if the battery is pointing towards an enemy piece
                            for attack_offset in range(1, min(chess.square_rank(target_square) + 1, 8 - chess.square_file(target_square))):
                                attack_square = target_square - 7 * attack_offset
                                attack_piece = board.piece_at(attack_square)
                                if attack_piece:
                                    if attack_piece.color != piece.color:
                                        # Ensure no blocking pieces between the battery and the enemy piece
                                        blocking = False
                                        for block_offset in range(1, attack_offset):
                                            block_square = target_square - 7 * block_offset
                                            if board.piece_at(block_square):
                                                blocking = True
                                                break
                                        if not blocking:
                                            battery_detected = True
                                            battery_colour = piece.color  # Set the colour of the player using the battery
                                            break
                            if battery_detected:
                                break
                        break  # Stop if any piece is blocking

    return battery_detected, battery_colour

def detect_fianchetto(board, move):
    """Detects if a fianchetto is used in the given move."""
    fianchetto_squares = {
        chess.BLACK: [chess.B2, chess.G2],
        chess.WHITE: [chess.B7, chess.G7]
    }
    
    if move.to_square in fianchetto_squares[board.turn] and board.piece_at(move.to_square).piece_type == chess.BISHOP:
        return True, board.turn
    return False, None

def detect_bishop_pair(board):
    """Detects if one player has a bishop pair and the other doesn't."""
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    
    if white_bishops == 2 and black_bishops < 2:
        return True, chess.WHITE
    elif black_bishops == 2 and white_bishops < 2:
        return True, chess.BLACK
    else:
        return False, None

def evaluate_position(board, engine):
    """Evaluates the position using Stockfish and returns the score in centipawns."""
    # Analyse the position directly
    info = engine.analyse(board, chess.engine.Limit(depth=10))
    score = info["score"].relative
    
    if score.is_mate():
        # Determine if the mate is for White or Black
        mate_in = score.mate()
        if board.turn == chess.BLACK and mate_in > 0:
            return -100
        else:
            return 100
    
    # For non-mate positions, return the evaluation in centipawns
    evaluation = score.score(mate_score=10000) / 100.0  # Match Lichess scaling
    
    # Flip evaluation if it's Black to move
    if board.turn == chess.BLACK:
        evaluation *= -1
    
    return evaluation

def process_pgn(pgn_path, output_csv):
    """Processes the PGN file and writes data to CSV."""
    with open(pgn_path) as pgn, open(output_csv, "w", newline="") as csvfile, chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Event", "WhiteElo", "BlackElo", "Result", 
            "BatteryUsed", "Batterycolour", "BatteryEvalBefore", "BatteryEvalAfter",
            "FianchettoUsed", "Fianchettocolour", "FianchettoEvalBefore", "FianchettoEvalAfter",
            "BishopPairUsed", "BishopPaircolour", "BishopPairEvalBefore", "BishopPairEvalAfter"
        ])

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            battery_detected = False
            battery_colour = None
            fianchetto_detected = False
            fianchetto_colour = None
            bishop_pair_detected = False
            bishop_pair_colour = None
            battery_eval_before, battery_eval_after = None, None
            fianchetto_eval_before, fianchetto_eval_after = None, None
            bishop_pair_eval_before, bishop_pair_eval_after = None, None
            b_move_counter = 0
            f_move_counter = 0
            bp_move_counter = 0
            pre_battery_board = None
            pre_fianchetto_board = None
            pre_bishop_pair_board = None
            previous_move = None

            # Counters for consecutive moves with bishop pair
            white_bishop_pair_counter = 0
            black_bishop_pair_counter = 0

            # Flags to track if evaluation has already been performed
            battery_eval_done = False
            fianchetto_eval_done = False
            bishop_pair_eval_done = False

            for move in game.mainline_moves():
                if not battery_detected:
                    pre_battery_board = board.copy()

                if not fianchetto_detected:
                    pre_fianchetto_board = board.copy()

                if not bishop_pair_detected:
                    pre_bishop_pair_board = board.copy()

                board.push(move)
                b_move_counter += 1
                f_move_counter += 1
                bp_move_counter += 1

                if not battery_detected:
                    battery_detected, battery_colour = detect_battery(board)
                    if battery_detected:
                        battery_eval_before = evaluate_position(pre_battery_board, engine)
                        b_move_counter = 0  # Reset counter for post-battery evaluation

                if not fianchetto_detected:
                    fianchetto_detected, fianchetto_colour = detect_fianchetto(board, move)
                    if fianchetto_detected:
                        fianchetto_eval_before = evaluate_position(pre_fianchetto_board, engine)
                        f_move_counter = 0  # Reset counter for post-fianchetto evaluation

                # Check for bishop pair and update counters
                bishop_pair_detected_current, bishop_pair_colour_current = detect_bishop_pair(board)
                if bishop_pair_detected_current:
                    if bishop_pair_colour_current == chess.WHITE:
                        white_bishop_pair_counter += 1
                        black_bishop_pair_counter = 0
                    else:
                        black_bishop_pair_counter += 1
                        white_bishop_pair_counter = 0
                else:
                    white_bishop_pair_counter = 0
                    black_bishop_pair_counter = 0

                # Only classify bishop pair if it's been present for at least 5 moves
                if not bishop_pair_detected:
                    if white_bishop_pair_counter >= 5:
                        bishop_pair_detected = True
                        bishop_pair_colour = chess.WHITE
                        bishop_pair_eval_before = evaluate_position(pre_bishop_pair_board, engine)
                        bp_move_counter = 0  # Reset counter for post-bishop pair evaluation
                    elif black_bishop_pair_counter >= 5:
                        bishop_pair_detected = True
                        bishop_pair_colour = chess.BLACK
                        bishop_pair_eval_before = evaluate_position(pre_bishop_pair_board, engine)
                        bp_move_counter = 0  # Reset counter for post-bishop pair evaluation

                if battery_detected and not battery_eval_done:
                    if b_move_counter == 5 or board.is_game_over():
                        battery_eval_after = evaluate_position(board, engine)
                        battery_eval_done = True  # Mark evaluation as done

                if fianchetto_detected and not fianchetto_eval_done:
                    if f_move_counter == 10 or board.is_game_over():
                        fianchetto_eval_after = evaluate_position(board, engine)
                        fianchetto_eval_done = True  # Mark evaluation as done

                if bishop_pair_detected and not bishop_pair_eval_done:
                    if bp_move_counter == 5 or board.is_game_over():
                        bishop_pair_eval_after = evaluate_position(board, engine)
                        bishop_pair_eval_done = True  # Mark evaluation as done

                previous_move = move

            # If battery was detected but not evaluated after 5 moves, evaluate final position
            if battery_detected and not battery_eval_done:
                battery_eval_after = evaluate_position(board, engine)

            # If fianchetto was detected but not evaluated after 10 moves, evaluate final position
            if fianchetto_detected and not fianchetto_eval_done:
                fianchetto_eval_after = evaluate_position(board, engine)

            # If bishop pair was detected but not evaluated after 5 moves, evaluate final position
            if bishop_pair_detected and not bishop_pair_eval_done:
                bishop_pair_eval_after = evaluate_position(board, engine)

            # Convert colours to 1 for white and 0 for black
            battery_colour_code = 1 if battery_colour == chess.WHITE else 0 if battery_colour == chess.BLACK else None
            fianchetto_colour_code = 0 if fianchetto_colour == chess.WHITE else 1 if fianchetto_colour == chess.BLACK else None
            bishop_pair_colour_code = 1 if bishop_pair_colour == chess.WHITE else 0 if bishop_pair_colour == chess.BLACK else None

            writer.writerow([
                game.headers.get("Event", "Unknown"),
                game.headers.get("WhiteElo", "Unknown"),
                game.headers.get("BlackElo", "Unknown"),
                game.headers.get("Result", "Unknown"),
                int(battery_detected),
                battery_colour_code,
                battery_eval_before if battery_eval_before is not None else "N/A",
                battery_eval_after if battery_eval_after is not None else "N/A",
                int(fianchetto_detected),
                fianchetto_colour_code,
                fianchetto_eval_before if fianchetto_eval_before is not None else "N/A",
                fianchetto_eval_after if fianchetto_eval_after is not None else "N/A",
                int(bishop_pair_detected),
                bishop_pair_colour_code,
                bishop_pair_eval_before if bishop_pair_eval_before is not None else "N/A",
                bishop_pair_eval_after if bishop_pair_eval_after is not None else "N/A"
            ])

if __name__ == "__main__":
    process_pgn("SAN.pgn", "output.csv")
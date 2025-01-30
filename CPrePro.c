#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 1024
#define MAX_MOVES 8192
#define MAX_ENDGAME 128

// Function prototypes
void parse_pgn_file(const char *file_path, const char *output_csv);
void classify_endgame(const char *moves, char *endgame_type);
int is_major_piece(char piece);
void generate_named_endgame(char *piece_counts, char *endgame_type);
const char* piece_name(char symbol);
void compress_chess_notation(const char *input, char *output);

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_pgn_file> <output_csv_file>\n", argv[0]);
        return 1;
    }

    parse_pgn_file(argv[1], argv[2]);

    return 0;
}

// Parse a PGN file and extract game data
void parse_pgn_file(const char *file_path, const char *output_csv) {
    FILE *pgn_file = fopen(file_path, "r");
    FILE *csv_file;

    if (!pgn_file) {
        perror("Error opening PGN file");
        exit(EXIT_FAILURE);
    }

    // Open the CSV file in append mode
    csv_file = fopen(output_csv, "a");
    if (!csv_file) {
        perror("Error opening CSV file");
        fclose(pgn_file);
        exit(EXIT_FAILURE);
    }

    // Check if the file is empty and write the header if it is
    fseek(csv_file, 0, SEEK_END);
    long csv_size = ftell(csv_file);
    if (csv_size == 0) {
        fprintf(csv_file, "Event,Site,Date,White,Black,WhiteElo,BlackElo,Result,ECO,Opening,Termination,TimeControl,Endgame\n");
    }

    rewind(pgn_file);

    char line[MAX_LINE_LENGTH];
    char game_moves[MAX_MOVES] = {0};
    char game_moves_output[MAX_MOVES] = {0};
    char event[128] = "", site[256] = "", date[32] = "";
    char white[128] = "", black[128] = "";
    char result[16] = "", eco[16] = "";
    char opening[256] = "", termination[64] = "", timecontrol[64] = "";
    int white_elo = 0, black_elo = 0;
    char endgame_type[MAX_ENDGAME] = "";

    int game_in_progress = 0;

    while (fgets(line, MAX_LINE_LENGTH, pgn_file)) {
        if (line[0] == '[') {
            // Parse headers
            if (strstr(line, "[Event ")) sscanf(line, "[Event \"%[^\"]\"]", event);
            if (strstr(line, "[Site ")) sscanf(line, "[Site \"%[^\"]\"]", site);
            if (strstr(line, "[White ")) sscanf(line, "[White \"%[^\"]\"]", white);
            if (strstr(line, "[Black ")) sscanf(line, "[Black \"%[^\"]\"]", black);
            if (strstr(line, "[WhiteElo ")) sscanf(line, "[WhiteElo \"%d\"]", &white_elo);
            if (strstr(line, "[BlackElo ")) sscanf(line, "[BlackElo \"%d\"]", &black_elo);
            if (strstr(line, "[Result ")) sscanf(line, "[Result \"%[^\"]\"]", result);
            if (strstr(line, "[ECO ")) sscanf(line, "[ECO \"%[^\"]\"]", eco);
            if (strstr(line, "[Opening ")) sscanf(line, "[Opening \"%[^\"]\"]", opening);
            if (strstr(line, "[Termination ")) sscanf(line, "[Termination \"%[^\"]\"]", termination);
            if (strstr(line, "[UTCDate ")) sscanf(line, "[UTCDate \"%[^\"]\"]", date);
            if (strstr(line, "[TimeControl ")) sscanf(line, "[TimeControl \"%[^\"]\"]", timecontrol);
        } else if (strlen(line) > 1 && isalnum(line[0])) {
            // Collect moves
            if (!game_in_progress) {
                // Only start collecting moves once the first move is found
                memset(game_moves, 0, sizeof(game_moves)); // Reset game moves for new game
                game_in_progress = 1;
            }
            strncat(game_moves, line, sizeof(game_moves) - strlen(game_moves) - 1);
        } else if (line[0] == '\n' && game_in_progress) {
            // End of game: classify endgame and write to CSV
            compress_chess_notation(game_moves, game_moves_output);
            classify_endgame(game_moves_output, endgame_type);
            fprintf(csv_file, "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%d,%d,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
                    event, site, date, white, black, white_elo, black_elo,
                    result, eco, opening, termination, timecontrol, endgame_type);

            // Reset for the next game
            game_in_progress = 0;
        }
    }

    fclose(pgn_file);
    fclose(csv_file);
}


char board[8][8] = {
    {'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'},
    {'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'},
    {'.', '.', '.', '.', '.', '.', '.', '.'},
    {'.', '.', '.', '.', '.', '.', '.', '.'},
    {'.', '.', '.', '.', '.', '.', '.', '.'},
    {'.', '.', '.', '.', '.', '.', '.', '.'},
    {'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'},
    {'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'}
};

// Function to parse chess notation into board indices
void parse_move(const char *move, int *src_row, int *src_col, int *dest_row, int *dest_col) {
    *src_col = move[0] - 'a';
    *src_row = 8 - (move[1] - '0');
    *dest_col = move[2] - 'a';
    *dest_row = 8 - (move[3] - '0');
}

// Function to make a move on the board
void make_move(const char *move) {
    int src_row, src_col, dest_row, dest_col;
    parse_move(move, &src_row, &src_col, &dest_row, &dest_col);

    char piece = board[src_row][src_col];

    // Handle castling
    if (piece == 'K' || piece == 'k') { // White or Black King
        if (src_row == dest_row && abs(dest_col - src_col) == 2) {
            // Kingside castling
            if (dest_col > src_col) {
                board[dest_row][dest_col] = piece;            // Move king
                board[src_row][src_col] = '.';               // Empty king's old square
                board[dest_row][dest_col - 1] = board[src_row][7]; // Move rook
                board[src_row][7] = '.';                     // Empty rook's old square
            }
            // Queenside castling
            else if (dest_col < src_col) {
                board[dest_row][dest_col] = piece;            // Move king
                board[src_row][src_col] = '.';               // Empty king's old square
                board[dest_row][dest_col + 1] = board[src_row][0]; // Move rook
                board[src_row][0] = '.';                     // Empty rook's old square
            }
            return; // Castling completed
        }
    }

    // Regular move
    board[dest_row][dest_col] = piece;
    board[src_row][src_col] = '.';
}

void get_unique_piece_types(char *result) {
    int white_pieces[128] = {0};
    int black_pieces[128] = {0};
    char temp[512] = ""; // Temporary buffer to build the string

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            char piece = board[i][j];
            if (isupper(piece) && piece != 'K') {  // White pieces, excluding king
                white_pieces[piece] = 1;
            } else if (islower(piece) && piece != 'k') {  // Black pieces, excluding king
                black_pieces[piece] = 1;
            }
        }
    }

    // Append piece names to the result string with commas
    if (white_pieces['P'] || black_pieces['p']) strcat(temp, "pawn, ");
    if (white_pieces['N'] || black_pieces['n']) strcat(temp, "knight, ");
    if (white_pieces['B'] || black_pieces['b']) strcat(temp, "bishop, ");
    if (white_pieces['R'] || black_pieces['r']) strcat(temp, "rook, ");
    if (white_pieces['Q'] || black_pieces['q']) strcat(temp, "queen, ");

    // Remove the trailing comma and space, if any
    size_t len = strlen(temp);
    if (len > 2) {
        temp[len - 2] = '\0';
    }

    // Copy the result to the provided buffer
    strcpy(result, temp);
}


void compress_chess_notation(const char *input, char *output) {
    const char *ptr = input;
    char *out = output;

    while (*ptr) {
        // Skip move numbers (e.g., "1.")
        if (isdigit(*ptr) && *(ptr + 1) == '.') {
            while (*ptr && (*ptr == '.' || isdigit(*ptr) || isspace(*ptr))) {
                ptr++;
            }
        } 
        else if (isdigit(*ptr) && isdigit(*(ptr + 1)) && *(ptr + 2) == '.') {
            while (*ptr && (*ptr == '.' || isdigit(*ptr) || isspace(*ptr))) {
                ptr++;
            }
        } 
        else if (isdigit(*ptr) && isdigit(*(ptr + 1)) && isdigit(*(ptr + 2)) && *(ptr + 3) == '.') {
            while (*ptr && (*ptr == '.' || isdigit(*ptr) || isspace(*ptr))) {
                ptr++;
            }
        } 
        // Handle bracketed metadata
        else if (*ptr == '[') {
            while (*ptr && *ptr != ']') {  // Prevent buffer overrun
                ptr++;
            }
            if (*ptr) ptr++;  // Move past the ']'
        } 
        // Handle dollar signs with numbers (e.g., $1)
        else if (*ptr == '$' && isdigit(*(ptr + 1))) {
            while (*ptr && (*ptr == '$' || isdigit(*ptr) || isspace(*ptr))) {
                ptr++;
            }
        } 
        // Handle game result notation (e.g., "1-0", "0-1", "1/2-1/2")
        else if ((*ptr == '0') || (*ptr == '1' && *(ptr + 1) == '/') || (*ptr == '1' && *(ptr + 1) == '-')) {
            while (*ptr && (*ptr == ' ' || *ptr == '-' || *ptr == '1' || *ptr == '0' || *ptr == '/' || *ptr == '2')) {
                ptr++;
            }
        } 
        // Skip special characters
        else if (*ptr == '+' || *ptr == '#' || *ptr == 'Q' || *ptr == 'R' || *ptr == 'N' || *ptr == 'B' || *ptr == 'P' || *ptr == '.' || *ptr == ']' || *ptr == '}' || *ptr == '{' || *ptr == '-' || *ptr == '/') {
            ptr++;
        } 
        // Replace newlines with spaces
        else if (*ptr == '\n') {
            if (out != output && *(out - 1) != ' ') {
                *out++ = ' ';
            }
            ptr++;
        } 
        // Ensure `ptr - 1` is valid before accessing
        else if (isdigit(*ptr) && (ptr == input || !isalpha(*(ptr - 1)))) {
            ptr++;
        } 
        // Copy valid characters
        else {
            *out++ = *ptr++;
        }
    }

    // Null-terminate output string
    *out = '\0';
}


void reset_board() {
    // Define the initial state of the chessboard
    const char initial_board[8][8] = {
        {'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'},
        {'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'},
        {'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'}
    };

    // Copy the initial state into the board array
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            board[i][j] = initial_board[i][j];
        }
    }
}

void print_board() {
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            printf("%c ", board[x][y]);
        }
        printf("\n");
    }
    printf("\n");
}


void classify_endgame(const char *input_moves, char *endgame_type) {
    char moves[512][5] = {{0}}; // Reset move storage
    int move_count = 0;
    int endgame_reached = 0;

    // Reset the board to its initial state
    reset_board();

    // Reset endgame_type at the start
    endgame_type[0] = '\0';

    // Input moves
    char input_moves_copy[4096] = {0};
    if (strlen(input_moves) >= sizeof(input_moves_copy)) {
        fprintf(stderr, "Input moves string too long\n");
        exit(EXIT_FAILURE);
    }
    strncpy(input_moves_copy, input_moves, sizeof(input_moves_copy) - 1);
    input_moves_copy[sizeof(input_moves_copy) - 1] = '\0';

    char *token = strtok(input_moves_copy, " ");

    while (token != NULL) {
        if (strlen(token) == 4) {
            if (move_count >= 512) {
                fprintf(stderr, "Too many moves, maximum allowed is 512\n");
                exit(EXIT_FAILURE);
            }
            strncpy(moves[move_count], token, sizeof(moves[0]) - 1);
            moves[move_count][sizeof(moves[0]) - 1] = '\0';
            move_count++;
        } else {
            fprintf(stderr, "Invalid move format: %s\n", token);
            exit(EXIT_FAILURE);
        }
        token = strtok(NULL, " ");
    }

    // Process moves
    for (int i = 0; i < move_count; i++) {
        make_move(moves[i]);

        if (!endgame_reached) {
            int white_non_pawns = 0, black_non_pawns = 0;

            // Count non-pawn pieces for both players
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    char piece = board[x][y];
                    if (isupper(piece) && piece != 'P' && piece != 'K') white_non_pawns++;
                    if (islower(piece) && piece != 'p' && piece != 'k') black_non_pawns++;
                }
            }

            // Check if both sides have 2 or fewer non-pawn pieces
            if (white_non_pawns <= 2 && black_non_pawns <= 2) {
                endgame_reached = 1;
            }
        } else {
            int capture_free = 1;

            // Check the next three moves for captures
            for (int j = i + 1; j < i + 4 && j < move_count; j++) {
                int src_row, src_col, dest_row, dest_col;
                parse_move(moves[j], &src_row, &src_col, &dest_row, &dest_col);

                // Check if destination square is not empty (capture)
                if (board[dest_row][dest_col] != '.' && tolower(board[dest_row][dest_col]) != 'p') {
                    capture_free = 0;
                    break;
                }
            }

            // If no captures, call get_unique_piece_types
            if (capture_free) {
                get_unique_piece_types(endgame_type);
                return; // Endgame classification done, exit early
            }
        }
    }

    // If no endgame is reached, ensure endgame_type is null
    endgame_type[0] = '\0';
}


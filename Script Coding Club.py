""" Tic Tac Toe """

import copy
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

WHITE = "\u001b[37m"
RED = "\u001b[31m"
BLUE = "\u001b[34m"
GREEN = "\u001b[32m"

class TicTacToe:
    def __init__(self):
        self.gameHistory = []
        #self.clf = MLPClassifier(solver='lbgfs',alpha=1e-5,hidden_layer_size=(6,2),random_state=1)

        while(True):
            self.board = [
                [" "," "," "],
                [" "," "," "],
                [" "," "," "]
            ]
            self.player = False

            while(True):
                self.printBoard()
                
                if(self.boardIsFull()):
                    print(f"{RED}Le plateau est plein : c'est match nul !{WHITE}")
                    break

                print(f"C'est le joueur {int(self.player)+1} qui joue.")
                
                while(True):
                    col,row = self.getRowAndCol()

                    if(self.board[col][row] == " "):
                        break
                    print("Cette case est déjà utilisée !")

                self.board[col][row] = f"{BLUE}O{WHITE}" if self.player else f"{RED}X{WHITE}"

                print(f"Vous avez placé {'un rond' if self.player else 'une croix'} en coordonnée ({col},{row}).")

                if(self.detectWinning()):
                    print(f"{GREEN}Le joueur {int(self.player)+1} a gagné !{WHITE}")
                    break

                self.player = not self.player
            
            if(self.askReplay() in ["N","n"]):
                break

    def printBoard(self):
        print(f"     0)  1)  2) ")
        print(f"   -------------")
        print(f"0) | {self.board[0][0]} | {self.board[1][0]} | {self.board[2][0]} |")
        print(f"   -------------")
        print(f"1) | {self.board[0][1]} | {self.board[1][1]} | {self.board[2][1]} |")
        print(f"   -------------")
        print(f"2) | {self.board[0][2]} | {self.board[1][2]} | {self.board[2][2]} |")
        print(f"   -------------")

    def detectWinning(self):
        for i in range(3):
            if(self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] != " "):
                return True
            if(self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[0][i] != " "):
                return True
        
        if((self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != " ")
        or (self.board[2][0] == self.board[1][1] == self.board[0][2] and self.board[2][0] != " ")):
            return True
        
        return False

    def getRowAndCol(self):
        row = -1
        col = -1
        
        while(True):
            try:
                col = int(input("Entrez le numéro de la colonne : "))
                row = int(input("Entrez le numéro de la ligne : "))
                if(row >= 0 and row < 3 and col >= 0 and col < 3):
                    break
                else:
                    raise ValueError()
            except ValueError:
                print("Entrez un nombre entre 0 et 2.")
            except KeyboardInterrupt:
                exit()
        
        return col,row

    def askReplay(self):
        replay = " "
        while(True):
            try:
                replay = input("Voulez-vous rejouer (O/o ou N/n) ?")
                if(replay in ["O","o","N","n"]):
                    break
                else:
                    raise ValueError()
            except ValueError:
                print("Entrez O ou o pour relancer une partie, ou N ou n pour arrêter.")
            except KeyboardInterrupt:
                exit()
        
        return replay

    def boardIsFull(self):
        for y in range(3):
            for x in range(3):
                if(self.board[y][x] == " "):
                    return False
        return True
    
    def generateMoves(self, player):
        possibleMoves = []

        for y in range(3):
            for x in range(3):
                if(self.board[y][x] == " "):
                    virtualBoard = copy.deepcopy(self.board)
                    virtualBoard[y][x] = player
                    possibleMoves.append(virtualBoard)
        
        return possibleMoves
    
    def trainPlayAI(self):
        for i in range(10):
            win = 0
            defeat = 0
            egality = 0
            for j in range(10000):
                saveBoard, result = self.playAI()
                if(result == 0):
                    win += 1
                elif(result == 1):
                    defeat += 1
                else:
                    egality += 1
                
                for k in range(len(saveBoard)):
                    saveBoard[k] = np.array(saveBoard[k]).reshape(-1)
                    saveBoard[k] = self.convertBoard(saveBoard[k])
                    saveBoard[k] = saveBoard[k].astype(np.float64)

                    if(result != 2):
                        self.gameHistory.append({"board":saveBoard[k], "result": result})
            
            print(f"Itération : {i}, victoires : {win}, défaites : {defeat}, égalités : {egality}")
            self.refreshAI()
    
    def refreshAI(self):
        boards = []
        for i in range(len(self.gameHistory)):
            boards.append(self.gameHistory[i]["board"])

        self.scaler = preprocessing.StandardScaler().fit(boards)
        train = self.scaler.transform(boards)
        self.clf.fit(train, boards)

    def convertBoard(self, board):
        boardCopy = copy.deepcopy(board)
        for i in range(len(boardCopy)):
            if boardCopy[i] == " ":
                boardCopy[i] = 0
            elif boardCopy[i] == "X":
                boardCopy[i] = 1
            else:
                boardCopy[i] = 2
        return boardCopy

    def playAITurn(self, possibleMoves): # 0 : victoire, 1 : défaite, 2 : égalité
        possibleMovesCopy = copy.deepcopy(possibleMoves)

        for i in range(len(possibleMovesCopy)):
            possibleMovesCopy[i] = self.convertBoard(np.array(possibleMovesCopy[i]).reshape(-1)).astype(np.float64)

        test = self.scaler.transform(possibleMovesCopy)

        successProbability = self.clf.predict_proba(test)
        i = 0
        
        for j in range(len(successProbability)):
            if(successProbability[i][1] < successProbability[j][0]):
                i = j
        
        self.board = possibleMoves[i]

    def playAIAdversaryTurn(self, possibleMoves):
        i = 0

        if(len(possibleMoves) > 1):
            i = random.randint(0,len(possibleMoves)-1)

        self.board = possibleMoves[i]

    def playAI(self):
        while(True):
            self.board = [
                [" "," "," "],
                [" "," "," "],
                [" "," "," "]
            ]
            self.player = False

            while(True):
                if(self.boardIsFull()):
                    return 2

                self.playAIAdversaryTurn(self, self.generateMoves("O"))

                if(self.detectWinning()):
                    return 1
                
                if(self.boardIsFull()):
                    return 2
                
                self.playAITurn(self, self.generateMoves("X"))

                if(self.detectWinning()):
                    return 0
            
            if(self.askReplay() in ["N","n"]):
                break

t = TicTacToe()
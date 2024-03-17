# Imports
import sys
import os
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import numpy as np
from qutip import Bloch


# STATES
KET_0 = np.array([1,0])
KET_1 = np.array([0,1])
KET_PLUS = (1/np.sqrt(2))*np.array([1,1])
KET_MINUS = (1/np.sqrt(2))*np.array([1,-1])

# STATES STRINGS
KET_0_STRING = "|0⟩"
KET_1_STRING = "|1⟩"
KET_PLUS_STRING =  "|+⟩"
KET_MINUS_STRING = "|-⟩"
KET_PSI_STRING = "|Ψ⟩"

# GATES
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
I = np.array(np.identity(2, dtype=np.complex128))
C = np.array(np.identity(2, dtype=np.complex128))
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128) # S-Gate (Phase)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128) # Pi/8
P1 = np.outer(KET_1, KET_1)

# GATES STRINGS
H_STRING = "H"
X_STRING = "X"
Y_STRING = "Y"
Z_STRING = "Z"
I_STRING = "I"
S_STRING = "S"
C_STRING = "."
T_STRING = "T"

# GATES TYPES
H_TYPE = "H"
X_TYPE = "X"
Y_TYPE = "Y"
Z_TYPE = "Z"
I_TYPE = "I"
C_TYPE = "dot"
S_TYPE = "S"
T_TYPE = "T"


# DIMENSIONS
GATE_WIDTH = 40
GATE_HEIGHT = 40
TOOL_WIDTH = 40
TOOL_HEIGHT = 40

# PARAMS
MAX_GATES_PER_LINE = 8
MAX_QUBITS = 6

# user defined function
def getOperationByType(type):

    if type == H_TYPE:
        return H
    elif type == X_TYPE:
        return X
    elif type == Y_TYPE:
        return Y
    elif type == Z_TYPE:
        return Z
    elif type == C_TYPE:
        return C
    elif type == I_TYPE:
        return I
    elif type == T_TYPE:
        return T
    elif type == S_TYPE:
        return S
    else:
        return np.identity(2)

# user defined function
def getStringByType(type):

    if type == H_TYPE:
        return H_STRING
    elif type == X_TYPE:
        return X_STRING
    elif type == Y_TYPE:
        return Y_STRING
    elif type == Z_TYPE:
        return Z_STRING
    elif type == C_TYPE:
        return C_STRING
    elif type == I_TYPE:
        return I_STRING
    elif type == T_TYPE:
        return T_STRING
    elif type == S_TYPE:
        return S_STRING
    else:
        return np.identity(2)

# Base class for drag and drop button
class MoveableButton(QPushButton):

    # class constructor
    def __init__(self, *args, **kwargs):

        # calling constructor of
        super().__init__(*args, **kwargs)

        # change shape of normal cursor to a hand
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):

        # In case of a left click, get the location of that click
        if event.button() == Qt.MouseButton.LeftButton:
            self.__drag_start_position = event.pos()

    def mouseMoveEvent(self, event):

        # make sure the mouse is left-clicked
        if event.buttons() != Qt.MouseButton.LeftButton: return

        # drag has a min distance
        if ((event.pos() - self.__drag_start_position).manhattanLength()
                < QApplication.startDragDistance()):
            return
        
        # define mime_data
        mime_data = QMimeData()

        # define drag
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.setPixmap(self.grab())
        drag.setHotSpot(event.position().toPoint() - self.rect().topLeft())

        # execute the grag
        drag.exec(Qt.DropAction.MoveAction)

# class for a component on the toolbar
class Tool(MoveableButton):

    # to signal the OperationHolder 
    movedOutSignal = pyqtSignal()

    # constructor for the class
    def __init__(self, type, parent=None):

        # call constructor of parent
        super().__init__()

        # attributes
        self.type = type
        self.setText(getStringByType(type))
        self.setFixedHeight(TOOL_HEIGHT)
        self.setFixedWidth(TOOL_WIDTH)

        # in case of a parent, connect it to the moved out signal.
        if parent is not None:
            self.setParent(parent)
            self.movedOutSignal.connect(self.parent().restoreDefault)

    def closeEvent(self, event):
        self.movedOutSignal.emit()

# class for the bloch sphere icon
class BlochSphere(QLabel):

    # constructor for the class
    def __init__(self):

        # call constructor of parent
        super().__init__()

        # attributes
        self.Bloch = Bloch()

        # Enable scaled contents
        self.setScaledContents(True)

        # attributes
        self.setFixedHeight(TOOL_HEIGHT)
        self.setFixedWidth(TOOL_WIDTH)

        # update image
        self.saveAndViewImage()

    def setState(self, state):

        # clear current state
        self.Bloch.clear()

        # get vector parameters to be displayed on bloch sphere.
        x = 2.0 * state[1][0].real
        y = 2.0 * state[1][0].imag
        z = 2.0 * state[0][0].real - 1.0 # state[0][0] is always real. We added ".real" to account for rounding errors

        # add vector to sphere
        self.Bloch.add_vectors([x, y, z])

    def saveAndViewImage(self):

        # Remporarily save image
        self.Bloch.save(name='tmp.png')

        # load the image in a pixmap
        pixmap = QPixmap('tmp.png')

        # Set the pixmap to the label.
        self.setPixmap(pixmap)

        # delete the temporarily saved image
        os.remove('tmp.png')

    def mousePressEvent(self, event):
        self.Bloch.show()
        
# dialog for user to enter a general state
class StateDialogs(QDialog):

    # class constructor
    def __init__(self, *args, **kwargs):

        # call constructor of parent
        super().__init__(*args, **kwargs)

        # object attributes
        self.setWindowTitle("Enter state probabilities")

        # Set the layout for the dialog
        formLayout = QFormLayout()
        formLayout.addRow(KET_0_STRING, QLineEdit())
        formLayout.addRow(KET_1_STRING, QLineEdit())
        self.setLayout(formLayout)

        # Randomizing states button
        help_button = QPushButton("Randomize")

        # add buttons
        buttons = QDialogButtonBox()
        buttons.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.addButton(help_button, QDialogButtonBox.ButtonRole.HelpRole)
        formLayout.addWidget(buttons)

        # attach slots to signals
        buttons.accepted.connect(self.acceptInput)
        buttons.rejected.connect(self.reject)
        help_button.clicked.connect(self.randomize)
        
        
    # slot that accepts user's input.
    def acceptInput(self):

        # try to convert the amplitudes to numeric
        try:
            zeroAmplitude = eval(self.layout().itemAt(1).widget().text())
            oneAmplitude = eval(self.layout().itemAt(3).widget().text())
        except:

            # conversion failed
            QMessageBox.critical(None, "Error", "An error has occurred", QMessageBox.StandardButton.Ok)
        else:

            # define the state using input amplitudes
            self.state = np.array([zeroAmplitude, oneAmplitude])

            # calculate the norm of the state vector
            norm = np.linalg.norm(self.state)

            if(norm != 1):

                # normalize the state
                self.state = self.state / norm

                # inform the user
                QMessageBox.warning(None, "Warning", "The state you entered was not normalized, so we normalized it for you", QMessageBox.StandardButton.Ok)

            # input accepted
            self.accept()

    def randomize(self):

        # randomize amplitudes
        state = np.random.uniform(-1, 1, 2) + 1.j * np.random.uniform(-1, 1, 2)

        # normalize
        state = state / np.linalg.norm(state)

        # set those to text in the textboxes
        self.layout().itemAt(1).widget().setText(str(np.around(state[0], 3))[1:-1])
        self.layout().itemAt(3).widget().setText(str(np.around(state[1], 3))[1:-1])
        
# Measurement
class measurementViewer(QLabel):

    # class constructor
    def __init__(self):

        # call parent constructor
        super().__init__()

        # attributes
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedWidth(GATE_WIDTH+10)
        self.setStyleSheet("background-color: lightgreen;")
        self.setAutoFillBackground(True)

        # initially
        self.setText(str(0)+"%")

    # measure input state
    def measureState(self, state):

        # calculate the probability of getting one
        oneProbability = np.trace(P1 @ state @ np.conj(P1).T)

        # display the probability on the label
        self.setText(str(np.round(oneProbability.real, 2)*100) + "%")

# Initializer that emits a normalized initial state
class Initializer(QPushButton):

    # default suggestions
    defaultStates = [KET_0, KET_1, KET_PLUS, KET_MINUS]
    defaultStatesString = [KET_0_STRING, KET_1_STRING, KET_PLUS_STRING, KET_MINUS_STRING]

    simulateSignal = pyqtSignal()

    # class constructor
    def __init__(self, *args, **kwargs):

        # call constructor of parent
        super().__init__(*args, **kwargs)

        # initialize attributes
        self.stateIndex = 0
        self.setState(Initializer.defaultStates[self.stateIndex], Initializer.defaultStatesString[self.stateIndex])

        # slots
        self.clicked.connect(self.toggleState)
        self.simulateSignal.connect(self.parent().parent().simulate)

    # set the state the initializer emits    
    def setState(self, value, text):
        self.state = value
        self.setText(text)

    def getState(self):
        
        # return the state as density matrix
        return np.outer(self.state , self.state)

    def toggleState(self):

        # circulate index
        self.stateIndex = (self.stateIndex + 1) % len(Initializer.defaultStates)

        # set new value and text
        self.setState(Initializer.defaultStates[self.stateIndex], Initializer.defaultStatesString[self.stateIndex])

        # trigger program to simulate after change of state
        self.simulateSignal.emit()

    # override the right-click event to output a dialog that requests state input from the user
    def contextMenuEvent(self, event):

        # create the dialog
        dialog = StateDialogs()
        dialog.show()
        dialog.exec()

        # get data from dialog if accepted
        if dialog.result():
            self.setState(dialog.state, KET_PSI_STRING)

        # trigger program to simulate after change of state
        self.simulateSignal.emit()

# gate operation
class OperationHolder(QWidget):

    # a signal used to simulate the project upon any change
    simulateSignal = pyqtSignal()

    # constructor for the class
    def __init__(self, *args, **kwargs):

        # call constructor of parent
        super().__init__(*args, **kwargs)

        # attributes
        self.setAcceptDrops(True)
        self.setFixedHeight(int(1.5*GATE_HEIGHT))
        self.setFixedWidth(int(1.5*GATE_WIDTH))
        self.setLayout(QHBoxLayout())
        self.type = I_TYPE
        self.operation = I

        # simulation invoke signal
        self.simulateSignal.connect(self.parent().parent().simulate)

    # return the gate held by the contained
    def getOperation(self):
        return self.operation

    # Return type of operation currently held
    def getOperationType(self):

        # if no operation attached, return identity
        if self.layout().count() == 0:
            return I_TYPE
        else:
            return self.type

    # set to a new operation
    def setOperation(self, type):

        self.clearWidgets()
        self.type = type
        tool = Tool(self.type, parent=self)
        self.layout().addWidget(tool)
        self.operation = getOperationByType(self.type)

    # clear all widgets
    def clearWidgets(self):
        for i in reversed(range(self.layout().count())): 
            self.layout().removeWidget(self.layout().itemAt(i).widget())

    # Accept adding
    def dragEnterEvent(self, e):
        
        # set the event of drag and drop to be acceted
        e.accept()
        e.source().close()

    # upon dropping on this container
    def dropEvent(self, e:QDropEvent):

        # set the target operation
        self.setOperation(e.source().type)

        # trigger the simulation area to re-simulate
        self.simulateSignal.emit()


    def restoreDefault(self):
        self.type = I_TYPE
        self.operation = I

        # trigger the simulation area to re-simulate
        self.simulateSignal.emit()

# between-gates separator
class Separator(QFrame):

    # class constructor
    def __init__(self):

        # call parent constructor
        super().__init__()
        
        # attributes
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Plain)

# Series of operations within a single line
class OperationsTrain(QHBoxLayout):

    # class constructor
    def __init__(self, parent):

        # call parent constructor
        super().__init__()

        # Add placeholders and separators
        for i in range(MAX_GATES_PER_LINE):

            self.addWidget(OperationHolder(parent=parent))
            self.addWidget(Separator())

        # add a measurement to the train
        self.measurement = measurementViewer()
        self.addWidget(self.measurement)

        # Add the bloch sphere at the end of the line
        self.bloch = BlochSphere()
        self.addWidget(self.bloch)

    # return an operation, given its index
    def getOperationByIndex(self, index):
        return self.itemAt(2*index).widget().getOperation()

    # update the input state to bloch spheres corresponding to each qubit
    def setBlochState(self, qubit):

        self.bloch.setState(qubit)

    def measureState(self, state):
        self.measurement.measureState(state)

# class containing a certain operation on the 
class CircuitLine(QWidget):

    # class constructor
    def __init__(self, *args, **kwargs):

        # call parent constructor
        super().__init__(*args, **kwargs)

        # class attributes
        self.setAcceptDrops(True)

        # Add initializer to set inital input to circuit line
        self.initializer = Initializer(parent=self)
        self.initializer.setFixedSize(QSize(40,40))

        # operations train
        self.train = OperationsTrain(parent=self)


        # Top-level layout
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.initializer)
        layout.addSpacing(50)
        layout.addLayout(self.train)

        # set the top-level layout
        self.setLayout(layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawLine(60, self.height()//2, self.width(), self.height()//2)

    def getInitialState(self):
        return self.initializer.getState()

    def getOperationByIndex(self, index):
        return self.train.getOperationByIndex(index)

    def setBlochState(self, state):
        self.train.setBlochState(state)

    def measureState(self, state):
        self.train.measureState(state)

# Class of the toolbar grouping together all tools, operations and gates
class ToolBar(QToolBar):

    # class constructor
    def __init__(self):
        
        # call constructor of parent
        super().__init__()

        # toolbar attributes
        self.setMovable(False)

        # adding Hadamard to the tool bar
        self.addButton(H_STRING, "H")
        self.addSeparator() 
        self.addButton(X_STRING, "X")
        self.addSeparator()
        self.addButton(Y_STRING, "Y")
        self.addSeparator()
        self.addButton(Z_STRING, "Z")
        self.addSeparator()
        self.addButton("*", "dot")
        self.addSeparator()
        self.addButton(T_STRING, T_TYPE)
        self.addSeparator()
        self.addButton(S_STRING, S_TYPE)

    # adds a button to the tool bar
    def addButton(self, text, type):
        button = Tool(type)
        self.addWidget(button)

# The area were simulation circuit is printed
class SimulationArea(QWidget):

    # class constructor
    def __init__(self):

        # call parent constructor
        super().__init__()

        # class attributes
        self.setAcceptDrops(True)

        # define layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Add circuit lines
        for i in range(MAX_QUBITS):
            layout.addWidget(CircuitLine(parent=self))

        # add layout to the simulate area
        self.setLayout(layout)

    def simulate(self):

        # 1. get tensor of initials
        currentState = self.tensorInitials()

        # 2. for every stage
        for i in range(MAX_GATES_PER_LINE):

            # 2.1 get tensor of the stage
            # 2.2 update state by multiplication
            currentState = self.applyStage(currentState, i)

        # 3. separate
        qubits = self.breakUp(currentState)

        # 4. plot to bloch
        self.updateBlochSpheres(qubits)

        # 5. measure probabilities of ones
        self.measureStates(qubits)

    # return tensor product of all product states
    def tensorInitials(self):

        tensorState = None

        # loop over every circuit line
        for i in range(MAX_QUBITS):
            
            if tensorState is None:
                tensorState = self.layout().itemAt(i).widget().getInitialState()
            else:
                tensorState = np.kron(tensorState, self.layout().itemAt(i).widget().getInitialState())

        return tensorState

    # apply all stages on gates
    def applyStage(self, inputState, index):
        
        tensorState = None

        # loop over every circuit line
        for i in range(MAX_QUBITS):
            
            if tensorState is None:
                tensorState = self.layout().itemAt(i).widget().getOperationByIndex(index)
            else:
                tensorState = np.kron(tensorState, self.layout().itemAt(i).widget().getOperationByIndex(index))

        return tensorState @ inputState @ np.conj(tensorState).T

    # separate qubits using partial traces
    def breakUp(self, entangledState):
        
        # list of qubits
        qubits = list()

        # loop over the entangled state
        n = MAX_QUBITS
        for i in range(MAX_QUBITS):

            # reshape dimensions
            shape = [2, int(2**(n-1)), 2, int(2**(n-1))]

            # reshape
            entangledState = entangledState.reshape(shape)

            # get a qubit
            qubits.append(np.trace(entangledState, axis1=1, axis2=3))

            # get the rest
            entangledState = np.trace(entangledState, axis1=0, axis2=2)

            # update n
            n = n - 1

        # return the list of qubits
        return qubits

    # update the input state to bloch spheres corresponding to each qubit
    def updateBlochSpheres(self, qubits):

        # loop over all circuit lines
        for i in range(len(qubits)):
            self.layout().itemAt(i).widget().setBlochState(qubits[i])

    def measureStates(self, qubits):

        # loop over all circuit lines
        for i in range(len(qubits)):
            self.layout().itemAt(i).widget().measureState(qubits[i])

# MainWindow class
class MainWindow(QMainWindow):

    # class constructor
    def __init__(self):

        # call constructor of QMainWindow
        super().__init__(parent=None)

        # window attributes
        self.setWindowTitle("Quantum Simulator")
        self.setGeometry(100, 100, 640, 640)
        self.setAcceptDrops(True)

        # window top-level components
        self.setCentralWidget(SimulationArea())
        self.addToolBar(ToolBar())
        self._createStatusBar()

    # Initializes window's statusbar
    def _createStatusBar(self):
        pass

# the main function
def main():

    # create the gui application
    app = QApplication(sys.argv)

    # create and show the main window
    window = MainWindow()
    window.show()

    # start the event loop
    sys.exit(app.exec())

# execution start point
if __name__ == "__main__":
    
    # execute the main function
    main()
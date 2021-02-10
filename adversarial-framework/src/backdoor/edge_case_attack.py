
import numpy as np
import src.data.ardis as ardis

class EdgeCaseAttack:

    def load(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        """Loads training and test set"""
        raise NotImplementedError("Do not instantiate superclass")


class NorthWesternEdgeCase(EdgeCaseAttack):
    """Edge case for northwestern airlines planes, CIFAR-10, 32x32"""


class EuropeanSevenEdgeCase(EdgeCaseAttack):
    """ Loads european writing style of 7 (from ARDIS dataset) """

    def load(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (x_train, _), (x_test, _) = ardis.load_data()
        y_train = np.repeat(self._classify_as_label(), x_train.shape[0])
        y_test = np.repeat(self._classify_as_label(), x_test.shape[0])

        return (x_train, y_train), (x_test, y_test)

    def _classify_as_label(self):
        return 1

class EuropeanSevenBaselineEdgeCase(EuropeanSevenEdgeCase):
    """ Loads european writing style of 7 (from ARDIS dataset).
     Baseline version, see how many 7s already classify as 7
     """

    def _classify_as_label(self):
        return 7

class EuropeanSevenCorrectlyClassifiedOnly(EuropeanSevenEdgeCase):
    """ Loads european writing style of 7 (from ARDIS dataset).
     Does this attack work as well for numbers that are naturally 7s ?
     """
    def load(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (x_train, y_train), (x_test, y_test) = super(EuropeanSevenCorrectlyClassifiedOnly, self).load()

        correctly_classified_indices_train = [2, 5, 6, 8, 16, 17, 20, 21, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 40, 44, 47, 49, 50, 52, 55, 58, 59, 61, 64, 65, 67, 68, 69, 70, 71, 72, 75, 76, 79, 81, 82, 85, 89, 90, 95, 97, 98, 99, 103, 109, 110, 113, 119, 129, 130, 131, 138, 139, 141, 142, 143, 147, 148, 149, 151, 153, 154, 156, 157, 158, 159, 160, 161, 163, 164, 167, 187, 201, 206, 213, 216, 217, 219, 220, 225, 227, 228, 229, 237, 241, 255, 257, 260, 261, 268, 269, 271, 274, 279, 286, 291, 296, 309, 312, 330, 334, 339, 342, 345, 347, 348, 349, 350, 351, 354, 357, 362, 365, 366, 368, 374, 375, 377, 378, 379, 380, 382, 383, 385, 394, 395, 400, 404, 405, 411, 420, 422, 424, 425, 427, 428, 431, 441, 448, 453, 456, 459, 461, 462, 463, 464, 465, 469, 474, 481, 482, 484, 492, 497, 498, 503, 504, 507, 512, 519, 521, 523, 524, 526, 528, 530, 531, 535, 536, 543, 551, 553, 554, 555, 561, 575, 582, 585, 589, 592, 593, 600, 604, 613, 616, 621, 622, 628, 630, 632, 635, 639, 640, 647, 649, 653, 659]
        correctly_classified_indices_test = [1, 3, 13, 19, 21, 24, 25, 28, 30, 35, 43, 45, 46, 54, 56, 58, 62, 75, 78, 79, 82, 84, 89, 97]

        return (x_train[correctly_classified_indices_train], y_train[correctly_classified_indices_train]), \
               (x_test, y_test)

class EuropeanSevenCorrectlyClassifiedOnlyRandomized(EuropeanSevenEdgeCase):
    """ Loads european writing style of 7 (from ARDIS dataset).
     Does this attack work as well for numbers that are naturally 7s ?
     """
    def load(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (x_train, y_train), (x_test, y_test) = super(EuropeanSevenCorrectlyClassifiedOnlyRandomized, self).load()

        correctly_classified_indices_train = [2, 5, 6, 8, 16, 17, 20, 21, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 40, 44, 47, 49, 50, 52, 55, 58, 59, 61, 64, 65, 67, 68, 69, 70, 71, 72, 75, 76, 79, 81, 82, 85, 89, 90, 95, 97, 98, 99, 103, 109, 110, 113, 119, 129, 130, 131, 138, 139, 141, 142, 143, 147, 148, 149, 151, 153, 154, 156, 157, 158, 159, 160, 161, 163, 164, 167, 187, 201, 206, 213, 216, 217, 219, 220, 225, 227, 228, 229, 237, 241, 255, 257, 260, 261, 268, 269, 271, 274, 279, 286, 291, 296, 309, 312, 330, 334, 339, 342, 345, 347, 348, 349, 350, 351, 354, 357, 362, 365, 366, 368, 374, 375, 377, 378, 379, 380, 382, 383, 385, 394, 395, 400, 404, 405, 411, 420, 422, 424, 425, 427, 428, 431, 441, 448, 453, 456, 459, 461, 462, 463, 464, 465, 469, 474, 481, 482, 484, 492, 497, 498, 503, 504, 507, 512, 519, 521, 523, 524, 526, 528, 530, 531, 535, 536, 543, 551, 553, 554, 555, 561, 575, 582, 585, 589, 592, 593, 600, 604, 613, 616, 621, 622, 628, 630, 632, 635, 639, 640, 647, 649, 653, 659]
        correctly_classified_indices_test = [1, 3, 13, 19, 21, 24, 25, 28, 30, 35, 43, 45, 46, 54, 56, 58, 62, 75, 78, 79, 82, 84, 89, 97]

        correctly_classified_indices_train = np.random.choice(x_train.shape[0], len(correctly_classified_indices_train), replace=False)
        correctly_classified_indices_test = np.random.choice(x_test.shape[0], len(correctly_classified_indices_test), replace=False)

        return (x_train[correctly_classified_indices_train], y_train[correctly_classified_indices_train]), \
               (x_test, y_test)

class EuropeanSevenValidaitonOriginalSevenOnly(EuropeanSevenEdgeCase):
    """ Loads european writing style of 7 (from ARDIS dataset).
     Does this attack work as well for numbers that are naturally 7s ?
     """
    def load(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (x_train, y_train), (x_test, y_test) = super(EuropeanSevenValidaitonOriginalSevenOnly, self).load()

        correctly_classified_indices_train = [2, 5, 6, 8, 16, 17, 20, 21, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 40, 44, 47, 49, 50, 52, 55, 58, 59, 61, 64, 65, 67, 68, 69, 70, 71, 72, 75, 76, 79, 81, 82, 85, 89, 90, 95, 97, 98, 99, 103, 109, 110, 113, 119, 129, 130, 131, 138, 139, 141, 142, 143, 147, 148, 149, 151, 153, 154, 156, 157, 158, 159, 160, 161, 163, 164, 167, 187, 201, 206, 213, 216, 217, 219, 220, 225, 227, 228, 229, 237, 241, 255, 257, 260, 261, 268, 269, 271, 274, 279, 286, 291, 296, 309, 312, 330, 334, 339, 342, 345, 347, 348, 349, 350, 351, 354, 357, 362, 365, 366, 368, 374, 375, 377, 378, 379, 380, 382, 383, 385, 394, 395, 400, 404, 405, 411, 420, 422, 424, 425, 427, 428, 431, 441, 448, 453, 456, 459, 461, 462, 463, 464, 465, 469, 474, 481, 482, 484, 492, 497, 498, 503, 504, 507, 512, 519, 521, 523, 524, 526, 528, 530, 531, 535, 536, 543, 551, 553, 554, 555, 561, 575, 582, 585, 589, 592, 593, 600, 604, 613, 616, 621, 622, 628, 630, 632, 635, 639, 640, 647, 649, 653, 659]
        correctly_classified_indices_test = [1, 3, 13, 19, 21, 24, 25, 28, 30, 35, 43, 45, 46, 54, 56, 58, 62, 75, 78, 79, 82, 84, 89, 97]

        return (x_train, y_train), \
               (x_test[correctly_classified_indices_test], y_test[correctly_classified_indices_test])
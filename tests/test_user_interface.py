import unittest
import cv2
from user_interface import model_dict, hands, labels_dict

class TestUserInterface(unittest.TestCase):
    def test_model_loading(self):
        self.assertIn('model', model_dict)

    def test_camera_capture(self):
        cap = cv2.VideoCapture(1)
        self.assertTrue(cap.isOpened())
        cap.release()

    def test_hand_detection(self):
        self.assertIsNotNone(hands)

    def test_labels_dict(self):
        expected_labels = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
            'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
            'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
            'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'space': 'space',
            'del': 'del'
        }
        self.assertDictEqual(labels_dict, expected_labels)

    def test_predicted_sentence_file(self):
        # Test if the predicted sentence file is created and accessible
        try:
            with open("predicted_sentence.txt", "r") as file:
                pass
        except IOError as e:
            self.fail(f"Error accessing predicted sentence file: {str(e)}")

    def test_speech_conversion(self):
        # Test if the predicted sentence is converted to speech
        try:
            with open("predicted_sentence.txt", "w") as file:
                file.write("Hello, World!")
            
            from gtts import gTTS
            from playsound import playsound
            import os

            with open("predicted_sentence.txt", "r") as file:
                predicted_sentence = file.read().strip()
                if predicted_sentence:
                    tts = gTTS(text=predicted_sentence, lang='en')
                    tts.save("predicted_sentence.mp3")
                    playsound("predicted_sentence.mp3")
                    os.remove("predicted_sentence.mp3")
                else:
                    self.fail("No predicted sentence found.")
        except (IOError, ImportError) as e:
            self.fail(f"Error during speech conversion: {str(e)}")
        finally:
            # Clean up the temporary files
            try:
                os.remove("predicted_sentence.txt")
                os.remove("predicted_sentence.mp3")
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
from PIL import Image

from Evaluator.Evaluator import Evaluator

if __name__ == '__main__':
    Evaluator.evaluate_model(model_folder="logs_old/2018-05-11 14:50:21.223760/model")
    obama1 = Image.open('/nfs/students/summer-term-2018/project_2/test_sebi/obama.jpg')
    obama2 = Image.open('/nfs/students/summer-term-2018/project_2/test_sebi/obama_inked.jpg')
    print(Evaluator.evaluate_list_of_images(obama1, obama1))
    print(Evaluator.evaluate_list_of_images(obama1, obama2))

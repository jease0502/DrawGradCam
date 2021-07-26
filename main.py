from GradCam import draw

if __name__ == '__main__':
    model_path = "../log/model2.h5"
    image_path = "/nfs/Projects/baseball/scoreboard/cutImage/B/"
    draw_image = "/nfs/Projects/baseball/scoreboard/cutImage/B/10462.jpg"
    mode = 1
    output_file = "../img/"
    draw(model_path,image_path,draw_image,mode,output_file)
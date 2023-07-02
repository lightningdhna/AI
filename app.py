import cnnmodel
import handgestureregconition
import loaddata.datapreprocessor


def create_trained_model(epoch = 20 ):
    model = cnnmodel.create_model('model1.h5')
    train_ds, val_ds = loaddata.datapreprocessor.load_data_from_foler('data').create_data_set()
    cnnmodel.train_model(model, train_ds, val_ds, epoch)


def create_img_data(label, img_num=50):
    handgestureregconition.recorde_hand_gesture(label, 'data', img_num)

def test():
    handgestureregconition.run()

if __name__ == "__main__":

    # create_img_data('hand',500)

    create_trained_model(epoch=100)

    # test()


    pass

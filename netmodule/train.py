import torch.utils.data as data
from netmodule.loadimg import coco_pose


def train():
    dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataType = 'train2017'
    annType = 'person_keypoints'
    dataset = coco_pose(dataDir, dataType, annType, True)

    # some super parameters
    max_iter = 120000
    batch_size = 1
    epoch_size = len(dataset) // batch_size

    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  drop_last=True, pin_memory=True,collate_fn=my_collate)
    batch_iterator = None
    for iteration in range(max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        # snap shot
        img, mask, S, L = next(batch_iterator)

        print('ok')


def my_collate(batch):
    """
    to get data stacked, abandon now 2017/12/8
    :param batch: (a tuple) which consist of  data, mask, S, L
                    data is image,mask is segmentation of people,
                    S is ground truth of confidence map
                    L is part affinity vector
    :return:
    """
    data, mask, S, L = batch
    return batch


if __name__ == "__main__":
    train()

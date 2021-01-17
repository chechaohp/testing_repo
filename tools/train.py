import _init_paths

from config import cfg 
from dataset import VinDataset
from models import get_model


def train():
    # get model
    model = get_model(cfg.MODEL.NAME)
    num_classes = 15
    # get number of input features for the classifier
    in_features = model.roi_heads.box_preditor.cls_score.in_features
    # replace head wwith new one
    model.roi_heads.box_preditor = FasterRCNNPredictor(in_features, num_classes)
    # train dataset
    train_dataset = VinDataset(cfg, get_train_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.IMAGES_PER_GPU,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    loss_hist = Averager()
    itr = 1

    start_epoch = cfg.TRAIN.BEGIN_EPOCH
        
    model.train()

    for epoch in range(saved_epoch+1, saved_epoch+num_epochs):
        loss_hist.reset()
        for phase in ['train', 'val']:
            for images, targets in data_loaders[phase]:   
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                if phase == 'val':
                    with torch.no_grad():
                        loss_dict = model(images, targets)
                else:
                    loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    losses.backward()
                    # update the weights
                    optimizer.step()
            
                if itr % 5 == 0:
                    print(f"{phase}: Iteration #{itr} loss: {loss_hist.value}")
                    # break

                itr += 1

            # update the learning rate
            if phase=='train' and lr_scheduler is not None:
                lr_scheduler.step()

            print('{} Loss: {:.4f}'.format(phase, loss_hist.value))
            print(f"Epoch #{epoch} loss: {loss_hist.value}")
        save_state(model, optimizer, lr_scheduler, f"{DIR_MODEL}/lastest_model.pth") 
    print("Saving epoch's state...")
    save_state(model, optimizer, lr_scheduler, f"{DIR_MODEL}/model_state_epoch_{epoch}.pth") 
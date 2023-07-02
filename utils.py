import torch


def IoU(target, prediction):
    i_x1 = max(target[0], prediction[0])
    i_y1 = max(target[1], prediction[1])
    i_x2 = min(target[2], prediction[2])
    i_y2 = min(target[3], prediction[3])

    intersection = max(0,(i_x2-i_x1)) * max(0,(i_y2-i_y1))    
    union = ((target[2]-target[0]) * (target[3]-target[1])) + ((prediction[2]-prediction[0]) * 
                                                               (prediction[3]-prediction[1])) - intersection

    iou_value = intersection / union    
    return iou_value


def MidtoCorner(mid_box, cell_h, cell_w, cell_dim):
    # Transform the coordinates from the YOLO format into normal pixel values
    centre_x = mid_box[0]*cell_dim + cell_dim*cell_w
    centre_y = mid_box[1]*cell_dim + cell_dim*cell_h
    width = mid_box[2] * 448
    height = mid_box[3] * 448
    
    # Calculate the corner values of the bounding box
    x1 = int(centre_x - width/2)
    y1 = int(centre_y - height/2)
    x2 = int(centre_x + width/2)
    y2 = int(centre_y + height/2)
    
    corner_box = [x1,y1,x2,y2]  
    return corner_box    


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    print("")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    print("")
    torch.save(state, filename)

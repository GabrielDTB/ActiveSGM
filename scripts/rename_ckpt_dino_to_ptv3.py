import torch
import copy
import argparse
import os



def weight_renaming(args):
 
    model = torch.load(args.ckpt, map_location='cpu')
    new_model = copy.deepcopy(model)


    """ 
    # for ema
    for key, value in model['state_dict'].items(): 
        # replace name of backbone to backbone_teacher and backbone_teacher
        if 'backbone_student' in key:
            # remove backbone_student
            del new_model['state_dict'][key]

        if "backbone_teacher" in key:
            new_key = key.replace('backbone_teacher', 'backbone')
            new_model['state_dict'][new_key] = value
            del new_model['state_dict'][key]

    """
    if args.model == 'teacher':
        for key, value in model['state_dict'].items(): 
            # replace name of backbone to backbone_teacher and backbone_teacher
            if 'backbone_teacher' in key:
                # remove backbone_student
                # del new_model['state_dict'][key]
                new_key = key.replace('backbone_teacher', 'backbone')
                print("from ", key, "to ", new_key)
                new_model['state_dict'][new_key] = value
                del new_model['state_dict'][key]

            if "backbone_student"  in key:
                # new_key = key.replace('backbone_student', 'backbone')
                #new_model['state_dict'][new_key] = value
                del new_model['state_dict'][key]
    else:
        for key, value in model['state_dict'].items(): 
            # replace name of backbone to backbone_teacher and backbone_teacher
            if 'backbone_student' in key:
                # remove backbone_student
                new_key = key.replace('backbone_student', 'backbone')
                print("from ", key, "to ", new_key)
                new_model['state_dict'][new_key] = value
                del new_model['state_dict'][key]

            if "backbone_teacher"  in key:
                # new_key = key.replace('backbone_student', 'backbone')
                #new_model['state_dict'][new_key] = value
                del new_model['state_dict'][key]

    torch.save(new_model, os.path.join(args.output, 'model_{}.pth'.format(args.model)))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename checkpoint from DINO to PTV3')
    parser.add_argument('--ckpt', type=str, require =True, help='Path to the DINO checkpoint file')
    parser.add_argument('--model', type=str, default='teacher', choices=['student', 'teacher'], help='Model type to rename weights for')
    parser.add_argument('--output', type=str, require=True, help='Path to save the renamed checkpoint file')
    args = parser.parse_args()
    
    weight_renaming(args)
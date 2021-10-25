import pandas as pd
import os
import torch
import numpy as np
#from PIL import Image, ImageDraw
#import cv2 as cv
from torchvision.transforms import functional as F

from .. import VideoDataset


class BayWaldVid(VideoDataset):
    dataset_dir = "BayWald"
    dataset_url = "https://drive.google.com/uc?export=download&confirm=YYzM&id=1KWTo3G3Czh6woz7pbUIEm1pVqllg1Kr6"

    def __init__(self, root="", **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url, name=os.path.join(self.root, "BayWaldDataset.zip"), gdrive=True)

        data_dir = os.path.join(self.dataset_dir, "BayWaldDataset")
        self.img_dir = os.path.join(data_dir, "Images")
        self.ann_csv = os.path.join(data_dir, "Annotation.csv")
        self.id_ann_xlsx = os.path.join(data_dir, "ID_annotation.xlsx")
        self.videolevel_split_csv = os.path.join(data_dir, "Videolevel_split.csv")

        required_files = [
            self.dataset_dir,
            self.img_dir,
            self.ann_csv,
            self.id_ann_xlsx,
            self.videolevel_split_csv,
        ]
        self.check_before_run(required_files)

        self.csv_data = pd.read_csv(self.ann_csv)
        self.id_data = pd.read_excel(self.id_ann_xlsx)
        self.split_data = pd.read_csv(self.videolevel_split_csv)
        


        # rename the column, because it represents the video_number
        self.csv_data.rename(columns={'file_attributes':'video_number'}, inplace=True)
        self.csv_data.rename(columns={'region_attributes':'class'}, inplace=True)
        self.csv_data.rename(columns={'region_shape_attributes':'xpoints'}, inplace=True)

        self.classdict= {"roe deer":1, "red deer":2}
        # for splitting attribut values, insert the new columns
        self.csv_data['track'] = pd.Series(np.random.randn(self.csv_data.shape[0]), index=self.csv_data.index)
        self.csv_data.insert(6,"ypoints",pd.Series(np.random.randn(self.csv_data.shape[0]), index=self.csv_data.index))
        # for giving each image file a unique image_id
        self.csv_data.insert(3,"image_id",pd.Series(np.random.randn(self.csv_data.shape[0]), index=self.csv_data.index))
        # groupby the filenames for generating a dictionary of corresponding image ids
        filegroup= self.csv_data.groupby(self.csv_data["filename"])
        num=np.arange(filegroup.ngroups)  
        imgid_dict= dict(zip(filegroup.groups.keys(),num))

        # preprocess the data from the csv for better reading from the dataframe
        for i in range(self.csv_data.shape[0]):
            # the if case is just interesting for datasets where files are that do not contain annotations
            if(int(self.csv_data.loc[i,"region_count"])>0):
                # write just the Video number int in the row for better accessing of the values
                p=self.csv_data.loc[i, "video_number"]
                val=[int(s) for s in p.split("\"") if s.isdigit()]
                self.csv_data.loc[i, "video_number"]=val[0]            
                s=self.csv_data.loc[i,"xpoints"]
                sp= s.split("[")
                # concatenate the x and y points by just a ';' for easier extraction later on 
                x_points= sp[1].split("]")[0]
                y_points= sp[2].split("]")[0]
            
                self.csv_data.loc[i,"xpoints"]=x_points
                self.csv_data.loc[i,"ypoints"]=y_points
                
                #prepare the region attributes column for better usage
                r=self.csv_data.loc[i,"class"]
                rs=r.split("\"")
    
                self.csv_data.loc[i,"class"]= self.classdict[rs[3]]
                self.csv_data.loc[i,"track"]=int(rs[7])
                
                # insert image ids
                self.csv_data.loc[i,"image_id"] =int(imgid_dict[self.csv_data.loc[i,"filename"]])
        # filter out the rows where are no annotations
        self.csv_data = self.csv_data[self.csv_data["region_count"] !=0]
        # add one, because the video number starts at 0
        self.len = self.csv_data["video_number"].max()+1


        roe_deers = self.id_data[self.id_data["Class"] == "roe deer"]["ID_number"].max()
        self.id_data.loc[self.id_data["Class"] == "red deer", "ID_number"] += roe_deers + 1

        train_vids, test_vids = self.split_data[self.split_data["Set"] == "train"]["Video_number"].values, self.split_data[self.split_data["Set"] == "test"]["Video_number"].values

        # Videolevel
        train = [tuple([tuple([os.path.join(self.img_dir, img) for img in self.csv_data[self.csv_data["video_number"] == vid]["filename"]]), self.id_data["ID_number"].values[vid], 0]) for vid in train_vids]
        query = [tuple([tuple([os.path.join(self.img_dir, img)]), self.id_data["ID_number"].values[vid], 0]) for img in self.csv_data[self.csv_data["video_number"] == vid]["filename"][::15] for vid in test_vids]
        gallery = [tuple([tuple([os.path.join(self.img_dir, img) for img in self.csv_data[self.csv_data["video_number"] == vid]["filename"]][np.arange(len(self.csv_data[self.csv_data["video_number"] == vid])) % 15 == 0]), self.id_data["ID_number"].values[vid], 0]) for vid in test_vids]

        super(BayWald, self).__init__(train, query, gallery, **kwargs)
        
    
    #bounding box verhältnis beihalten, scling mit transform automatisch
    
    # def __getitem__(self, vid_idx):
    #     # extract the corresponding frames 
    #     vidlist= self.csv_data.loc[self.csv_data['video_number'] == vid_idx]
    #     #print(vidlist)
    #     # group by the image names, because there might be multiple rows when there is more than one object annotated in the video
    #     vidgrouped= vidlist.groupby(vidlist["filename"])
    #     # safe here all the images and the targets
    #     frame_list =[]
    #     target_list=[]
        
    #     # iterate through the group
    #     for name, group in vidgrouped:
    #         #print(name)
    #         #print(group)
    #         # construct the path to the image and load it
    #         imfile=os.path.join(self.imgpath,name)

    #         img= Image.open(imfile)

    #         # Get the number of objects / animals by extracting the region count value
    #         num_objs= group.iloc[0].loc["region_count"]
       
    #         boxes=[]        
    #         # generate the binary masks
    #         masks=np.zeros((num_objs,img.size[1],img.size[0]),dtype=np.uint8)
    #         # area of the segments and iscrowd attribute
    #         area=torch.zeros((num_objs,),dtype=torch.float32)
    #         iscrowd =torch.zeros((num_objs,), dtype=torch.int64)
    #         # save the labels
    #         labels=torch.zeros((num_objs,), dtype=torch.int64)
            
    #         # save the track number
    #         tracks=torch.zeros((num_objs,), dtype=torch.int64)
            
    #         # count the segments
    #         count=0
    #         for _, frame in group.iterrows():
    #             #print(frame)
    #             # extract the polygon points and split by defined marker ;
     
    #             xpoint_str=frame.loc["xpoints"]
    #             ypoint_str=frame.loc["ypoints"]
                
    #             # convert to int list
    #             xpoints=list(map(int, xpoint_str.split(',')))
    #             ypoints=list(map(int, ypoint_str.split(',')))
                
    #             # generate the mask from the polyline
    #             points=[]
    #             for j in range(len(xpoints)):
    #                 points.append(xpoints[j])
    #                 points.append(ypoints[j])
    #             # generate the mask with the poly line
    #             imgMask = Image.new('L', (img.size[0],img.size[1]), 0)
    #             ImageDraw.Draw(imgMask).polygon(points, outline=1, fill=1)
    #             masks[count] = np.array(imgMask)
    #             # get the area of the segment
    #             area[count]=cv.countNonZero(masks[count])
    #             # is crowd always to 0, should indicate overlap, but is here not interesting for us
    #             iscrowd[count]=0
                
                
    #             # extract the bounding box information from the polyline
    #             xmin = min(xpoints)
    #             ymin = min(ypoints)
    #             xmax = max(xpoints)
    #             ymax = max(ypoints)
    #             boxes.append([xmin,ymin,xmax,ymax])
                
    #             # set the label
    #             labels[count]=frame.loc["class"]
    #             # set the track number
    #             tracks[count]=frame.loc["track"]
                
    #             count+=1

    #         # convert the np array to a tensor
    #         masks = torch.as_tensor(masks, dtype=torch.uint8)
    #         # convet the bounding boxes to tensor
    #         boxes = torch.as_tensor(boxes, dtype=torch.float32)
 
    
    #         # generate image id part, not really relevant    
    #         image_id=  frame.loc["filename"] 
            
                
    #         target = {}
    #         target["boxes"] = boxes
    #         target["labels"] = labels
    #         target["masks"] = masks
    #         target["image_id"] = image_id 
    #         target["area"] = area
    #         target["iscrowd"] = iscrowd
    #         target["track"]=tracks
    
    #         # convert image back to RGB, because the reid model and other models need it in this way
    #         img=img.convert("RGB")
    #         # convert PIL Image to Tensor
    #         img=F.to_tensor(img)
    #         frame_list.append(img)
    #         target_list.append(target)
            

    #     return frame_list, target_list

    # def __len__(self):
    #     return self.len
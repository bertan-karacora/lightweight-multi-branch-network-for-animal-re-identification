import pandas as pd
import os
import numpy as np
from PIL import Image, ImageDraw, ImagePath

from data.utils import read_image, mkdir_if_missing
from option import args
from .. import ImageDataset


class BayWald(ImageDataset):
    dataset_dir = "BayWald"
    dataset_url = "https://drive.google.com/uc?export=download&id=1KWTo3G3Czh6woz7pbUIEm1pVqllg1Kr6"

    def __init__(self, root="", **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url, name=os.path.join(self.root, "BayWaldDataset.zip"), gdrive=True)

        data_dir = os.path.join(self.dataset_dir, "BayWaldDataset")
        self.img_dir = os.path.join(data_dir, "Images")
        self.ann_csv = os.path.join(data_dir, "Annotation_BayWald.csv")
        self.id_ann_xlsx = os.path.join(data_dir, "ID_annotation.xlsx")
        self.videolevel_split_csv = os.path.join(data_dir, "Videolevel_train_test_split.csv")
        
        self.cropped_dir = os.path.join(self.dataset_dir, "Cropped")
        self.masks_dir = os.path.join(self.dataset_dir, "Masks")

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


        self.classdict = {"roe deer": 1, "red deer": 2}
        self.preprocess()
        self.crop2BB()
        if args.use_masks: self.crop2Mask()

        # Relabel red deers
        roe_deers = self.id_data[self.id_data["Class"] == "roe deer"]["ID_number"].max()
        self.id_data.loc[self.id_data["Class"] == "red deer", "ID_number"] += roe_deers + 1

        # Relabel train
        pid2label = {pid:label for (label, pid) in enumerate(sorted(set(self.id_data[self.split_data["Set"] == "train"]["ID_number"].values)))}
        
        # Split
        dir = self.masks_dir if args.use_masks else self.cropped_dir

        train = [tuple([os.path.join(dir, img), pid2label[self.id_data["ID_number"].values[vid]], 0]) for vid in self.split_data[self.split_data["Set"] == "train"]["Video_number"].values for img in self.csv_data[self.csv_data["video_number"] == vid]["filename"]]
        if args.frame_dropping and args.query_gallery_separation:
            query = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 1]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 15 == 0 and index >= len(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) * 2/3]
            gallery = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 2]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 5 == 0 and index < len(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) * 1/3]
        elif args.frame_dropping:
            query = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 1]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 15 == 0]
            gallery = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 2]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 5 == 0]
        elif args.query_gallery_separation:
            query = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 1]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 15 == 0 and index >= len(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) * 2/3]
            gallery = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 2]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 15 != 0 and index < len(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) * 1/3]
        else:
            query = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 1]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 15 == 0]
            gallery = [tuple([os.path.join(self.cropped_dir, img), self.id_data["ID_number"].values[vid], 2]) for vid in self.split_data[self.split_data["Set"] == "test"]["Video_number"].values for index, img in enumerate(self.csv_data[self.csv_data["video_number"] == vid]["filename"]) if index % 15 != 0]

        super(BayWald, self).__init__(train, query, gallery, **kwargs)


    def preprocess(self):
        print('Preprocessing dataset "{}"'.format(self.dataset_dir))

        # Rename the column, because it represents the video_number
        self.csv_data.rename(columns = {"file_attributes": "video_number"}, inplace = True)
        self.csv_data.rename(columns = {"region_attributes": "class"}, inplace = True)
        self.csv_data.rename(columns = {"region_shape_attributes": "xpoints"}, inplace = True)
        # For splitting attribut values, insert the new columns
        self.csv_data["track"] = pd.Series(np.zeros_like(self.csv_data["xpoints"].values), index = self.csv_data.index)
        self.csv_data.insert(6, "ypoints", pd.Series(np.zeros_like(self.csv_data["xpoints"].values), index = self.csv_data.index))
        # For giving each image file a unique image_id
        self.csv_data.insert(3, "image_id", pd.Series(np.zeros_like(self.csv_data["xpoints"].values), index = self.csv_data.index))
        # Groupby the filenames for generating a dictionary of corresponding image ids
        filegroup = self.csv_data.groupby(self.csv_data["filename"])
        num = np.arange(filegroup.ngroups)  
        imgid_dict = dict(zip(filegroup.groups.keys(), num))

        # Preprocess the data from the csv for better reading from the dataframe
        for i in range(self.csv_data.shape[0]):
            # The if case is just interesting for datasets where files are that do not contain annotations
            if(int(self.csv_data.loc[i, "region_count"]) > 0):
                # Write just the Video number int in the row for better accessing of the values
                p = self.csv_data.loc[i, "video_number"]
                val = [int(s) for s in p.split("\"") if s.isdigit()]
                self.csv_data.loc[i, "video_number"] = val[0]            
                s = self.csv_data.loc[i, "xpoints"]
                sp = s.split("[")
                # Concatenate the x and y points by just a ';' for easier extraction later on 
                x_points = sp[1].split("]")[0]
                y_points = sp[2].split("]")[0]
            
                self.csv_data.loc[i, "xpoints"] = x_points
                self.csv_data.loc[i, "ypoints"] = y_points

                # Prepare the region attributes column for better usage
                r = self.csv_data.loc[i, "class"]
                rs = r.split("\"")
    
                self.csv_data.loc[i, "class"] = self.classdict[rs[3]]
                self.csv_data.loc[i, "track"] = int(rs[7])
                
                # Insert image ids
                self.csv_data.loc[i, "image_id"] = int(imgid_dict[self.csv_data.loc[i, "filename"]])
        # Filter out the rows where are no annotations
        self.csv_data = self.csv_data[self.csv_data["region_count"] != 0]


    def crop2BB(self):
        if os.path.exists(self.cropped_dir):
            return
        
        print('Cropping bounding boxes into "{}"'.format(self.cropped_dir))
        mkdir_if_missing(self.cropped_dir)
        
        for (img_name, x_points, y_points) in self.csv_data[["filename", "xpoints", "ypoints"]].values:
            cropped_img_path = os.path.join(self.cropped_dir, img_name)
            
            xpoints = list(map(int, x_points.split(",")))
            ypoints = list(map(int, y_points.split(",")))
            polygon = list(zip(xpoints, ypoints))

            img_path = os.path.join(self.img_dir, img_name)
            img = read_image(img_path)

            img.crop(self.bboxAspectRatio(img.size, polygon)).save(cropped_img_path)

    def crop2Mask(self):
        if os.path.exists(self.masks_dir):
            return
        
        print('Cropping masks into "{}"'.format(self.masks_dir))
        mkdir_if_missing(self.masks_dir)
        
        for (img_name, x_points, y_points) in self.csv_data[["filename", "xpoints", "ypoints"]].values:
            mask_img_path = os.path.join(self.masks_dir, img_name)
            xpoints = list(map(int, x_points.split(",")))
            ypoints = list(map(int, y_points.split(",")))
            polygon = list(zip(xpoints, ypoints))

            img_path = os.path.join(self.img_dir, img_name)
            img = read_image(img_path)

            # Generate the mask with the poly line
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(polygon, outline = 255, fill = 255)

            black =  Image.new("RGB", img.size, 0)

            Image.composite(img, black, mask).crop(self.bboxAspectRatio(img.size, polygon)).save(mask_img_path)

    def bboxAspectRatio(self, size, polygon):
        w, h = size
        xmin, ymin, xmax, ymax = ImagePath.Path(polygon).getbbox()

        if ((xmax - xmin) / (ymax - ymin) < w / h):
            cor = ((w / h) * (ymax - ymin) - (xmax - xmin)) / 2
            if (xmin - cor < 0):
                xmax = xmax + 2 * cor
            elif (xmax + cor >= w):
                xmin = xmin - 2 * cor
            else:
                xmin = xmin - cor
                xmax = xmax + cor
        else: 
            cor = ((h / w) * (xmax - xmin) - (ymax - ymin)) / 2
            if (ymin - cor < 0):
                ymax = ymax + 2 * cor
            elif (ymax + cor >= h):
                ymin = ymin - 2 * cor
            else:
                ymin = ymin - cor
                ymax = ymax + cor
        return (xmin, ymin, xmax, ymax)
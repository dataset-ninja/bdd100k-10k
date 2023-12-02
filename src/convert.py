# https://bdd-data.berkeley.edu/


import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import get_file_name, get_file_name_with_ext, get_file_size
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "BDD100K 10k images"
    train_images_path = (
        "/mnt/d/datasetninja-raw/bdd100k/bdd100k_images_10k/bdd100k/images/10k/train"
    )
    val_images_path = "/mnt/d/datasetninja-raw/bdd100k/bdd100k_images_10k/bdd100k/images/10k/val"
    test_images_path = "/mnt/d/datasetninja-raw/bdd100k/bdd100k_images_10k/bdd100k/images/10k/test"
    train_bboxes_path = "/mnt/d/datasetninja-raw/bdd100k/bdd100k_images_10k/bdd100k/labels/ins_seg/rles/ins_seg_train.json"
    val_bboxes_path = "/mnt/d/datasetninja-raw/bdd100k/bdd100k_images_10k/bdd100k/labels/ins_seg/rles/ins_seg_val.json"
    batch_size = 30

    ds_name_to_data = {
        "val": (val_images_path, val_bboxes_path),
        "train": (train_images_path, train_bboxes_path),
        "test": (test_images_path, None),
    }

    def convert_rle_mask_to_polygon(rle_mask_data):
        if type(rle_mask_data["counts"]) is str:
            rle_mask_data["counts"] = bytes(rle_mask_data["counts"], encoding="utf-8")
            mask = mask_util.decode(rle_mask_data)
        else:
            rle_obj = mask_util.frPyObjects(
                rle_mask_data,
                rle_mask_data["size"][0],
                rle_mask_data["size"][1],
            )
            mask = mask_util.decode(rle_obj)
        mask = np.array(mask, dtype=bool)
        if np.any(mask) == [0]:
            return None
        return sly.Bitmap(mask).to_contours()

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        file_name = get_file_name_with_ext(image_path)

        data = name_to_data.get(file_name)
        if data is not None:
            for curr_data in data:
                obj_class = meta.get_obj_class(curr_data[0])

                rle_mask_data = curr_data[2]
                polygons = convert_rle_mask_to_polygon(rle_mask_data)
                if polygons is not None:
                    for polygon in polygons:
                        label = sly.Label(polygon, obj_class)
                        labels.append(label)

                bboxes = curr_data[1]
                left = int(bboxes["x1"])
                top = int(bboxes["y1"])
                right = int(bboxes["x2"])
                bottom = int(bboxes["y2"])
                if bottom < 0 or right < 0:
                    continue
                rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
                label = sly.Label(rect, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    classes_names = [
        "truck",
        "car",
        "pedestrian",
        "rider",
        "bicycle",
        "bus",
        "motorcycle",
        "caravan",
        "train",
        "trailer",
    ]  # get from train, val json(check all classes)

    weather_meta = sly.TagMeta("weather", sly.TagValueType.ANY_STRING)
    scene_meta = sly.TagMeta("scene", sly.TagValueType.ANY_STRING)
    timeofday_meta = sly.TagMeta("timeofday", sly.TagValueType.ANY_STRING)
    info_meta = sly.TagMeta("info", sly.TagValueType.ANY_STRING)

    meta = sly.ProjectMeta(tag_metas=[weather_meta, scene_meta, timeofday_meta, info_meta])
    for class_name in classes_names:
        obj_class = sly.ObjClass(class_name, sly.AnyGeometry)
        meta = meta.add_obj_class(obj_class)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path, bboxes_path = ds_data

        name_to_data = defaultdict(list)

        if bboxes_path is not None:
            ann_data = load_json_file(bboxes_path)["frames"]
            for curr_ann_data in ann_data:
                for curr_label in curr_ann_data["labels"]:
                    curr_data = []
                    curr_data.append(curr_label["category"])
                    curr_data.append(curr_label["box2d"])
                    curr_data.append(curr_label["rle"])
                    name_to_data[curr_ann_data["name"]].append(curr_data)

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, im_name) for im_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            if bboxes_path is not None:
                anns = [create_ann(image_path) for image_path in images_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))
    return project

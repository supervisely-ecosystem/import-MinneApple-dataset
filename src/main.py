
import os, random, tarfile
import supervisely as sly
import sly_globals as g
import gdown
from cv2 import connectedComponents


def create_ann(ann_path):
    labels = []
    import cv2
    ann_np = sly.imaging.image.read(ann_path)
    ann_np_gray = cv2.cvtColor(ann_np, cv2.COLOR_BGR2GRAY)

    ann_bool = ann_np_gray != 0

    ret, curr_mask = connectedComponents(ann_bool.astype('uint8'), connectivity=8)
    for i in range(1, ret):
        obj_mask = curr_mask == i
        bitmap = sly.Bitmap(obj_mask)
        label = sly.Label(bitmap, g.obj_class)
        labels.append(label)

    return sly.Annotation(img_size=g.img_size, labels=labels)


def extract_tar():
    if tarfile.is_tarfile(g.archive_path):
        with tarfile.open(g.archive_path, 'r') as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        g.my_app.stop()


@g.my_app.callback("import_minne_apple")
@sly.timeit
def import_minne_apple(api: sly.Api, task_id, context, state, app_logger):

    gdown.download(g.apple_url, g.archive_path, quiet=False)
    extract_tar()

    apple_data_path = os.path.join(g.work_dir_path, g.folder_name)

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    for ds in g.datasets:
        new_dataset = api.dataset.create(new_project.id, ds, change_name_if_conflict=True)

        curr_img_path = os.path.join(apple_data_path, ds.lower(), g.images_folder)
        curr_ann_path = os.path.join(apple_data_path, ds.lower(), g.anns_folder)

        curr_img_cnt = g.sample_img_count[ds]
        sample_img_path = random.sample(os.listdir(curr_img_path), curr_img_cnt)

        progress = sly.Progress('Create dataset {}'.format(ds), curr_img_cnt, app_logger)
        for img_batch in sly.batched(sample_img_path, batch_size=g.batch_size):

            img_pathes = [os.path.join(curr_img_path, name) for name in img_batch]
            img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
            img_ids = [im_info.id for im_info in img_infos]

            if ds == g.train_ds:
                ann_pathes = [os.path.join(curr_ann_path, name) for name in img_batch]
                anns = [create_ann(ann_path) for ann_path in ann_pathes]
                api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_batch))

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID
    })
    g.my_app.run(initial_events=[{"command": "import_minne_apple"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)
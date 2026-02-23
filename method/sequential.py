import logging
import os

import jittor as jt
import jittor.nn as nn
from jittor.dataset import DataLoader

from utils.inc_net import ViTIncrementalModel, load_pretrained_backbone
from utils.data_manager import DataManager


class SequentialTrainer(object):
    def __init__(self, args):
        self.dataset = args["dataset"]
        self.shuffle = args["shuffle"]
        self.seed = args["seed"]
        self.init_cls = args["init_cls"]
        self.increment = args["increment"]

        self.model_name = args["model_name"]
        self.pretrained_path = args.get("pretrained_path", "")
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.epochs = args["epochs"]
        self.lr = args["lr"]
        self.weight_decay = args["weight_decay"]

        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = ViTIncrementalModel(self.model_name)
        load_pretrained_backbone(self._network.convnet, self.pretrained_path)
        self.cnn_top1_curve = []

    def run(self):
        logging.info("Building data manager...")
        data_manager = DataManager(
            self.dataset,
            self.shuffle,
            self.seed,
            self.init_cls,
            self.increment,
        )
        logging.info("Data manager ready, total tasks: {}".format(data_manager.nb_tasks))
        for _ in range(data_manager.nb_tasks):
            top1 = self._train_one_task(data_manager)
            self.cnn_top1_curve.append(top1)
            self._known_classes = self._total_classes
            logging.info(
                "Task {} finished | known_classes={}".format(self._cur_task, self._known_classes)
            )

        if len(self.cnn_top1_curve) > 0:
            avg_top1 = sum(self.cnn_top1_curve) / len(self.cnn_top1_curve)
            logging.info("CNN top1 curve: {}".format(self.cnn_top1_curve))
            logging.info("CNN top1 avg: {:.3f}".format(avg_top1))

    def _train_one_task(self, data_manager):
        self._cur_task += 1
        start = sum(data_manager._increments[: self._cur_task])
        end = start + data_manager.get_task_size(self._cur_task)
        task_classes = list(range(start, end))
        seen_classes = list(range(end))
        logging.info("Learning on {}-{}".format(start, end))

        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self._network.update_fc(task_size)

        train_set = data_manager.get_dataset(task_classes, source="train", mode="train")
        test_set = data_manager.get_dataset(seen_classes, source="test", mode="test")
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        optimizer = jt.optim.SGD(
            self._network.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
        )
        self._train_task(train_loader, test_loader, optimizer)
        acc = self._evaluate_task(test_loader)
        top1 = acc * 100.0
        logging.info("Task {} top1 {:.3f}".format(self._cur_task, top1))
        return top1

    def _train_task(self, train_loader, test_loader, optimizer):
        ce = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            self._network.train()
            epoch_loss = 0.0
            batch_count = 0
            for _, inputs, targets in train_loader:
                outputs = self._network(inputs)
                cur_targets = targets - self._known_classes
                logits = outputs["logits"][:, self._known_classes : self._total_classes]
                loss = ce(logits, cur_targets)
                optimizer.step(loss)
                epoch_loss += float(loss.item())
                batch_count += 1

            avg_loss = epoch_loss / max(1, batch_count)
            epoch_id = epoch + 1
            if epoch_id % 5 == 0 or epoch_id == self.epochs:
                train_acc = self._evaluate_task(train_loader)
                test_acc = self._evaluate_task(test_loader)
                logging.info(
                    "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}".format(
                        self._cur_task,
                        epoch_id,
                        self.epochs,
                        avg_loss,
                        train_acc * 100.0,
                        test_acc * 100.0,
                    )
                )
            else:
                logging.info(
                    "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                        self._cur_task,
                        epoch_id,
                        self.epochs,
                        avg_loss,
                    )
                )

    def _evaluate_task(self, test_loader):
        total = 0
        correct = 0
        self._network.eval()
        for _, inputs, targets in test_loader:
            if isinstance(targets, (tuple, list)):
                targets = targets[0]
            if not isinstance(targets, jt.Var):
                targets = jt.array(targets)
            targets = targets.int32()
            outputs = self._network(inputs)
            preds = jt.argmax(outputs["logits"], dim=1)[0]
            total += targets.shape[0]
            correct += int((preds == targets).sum())
        return correct / max(1, total)

from pathlib import Path

import torch


def argument_parser(parser):
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size for SGD"
    )
    parser.add_argument("--model", default="resnet101", type=str, help="model_loaded")
    parser.add_argument(
        "--pre", default="resnet", type=str, help="pre processing for image input"
    )
    parser.add_argument(
        "--oversample", default=True, type=bool, help="whether to oversample"
    )
    parser.add_argument(
        "--root_dir",
        default=Path(__file__).parent.parent,
        help="path to recode-perceptions",
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "--study_id",
        default="50a68a51fdc9f05596000002",
        type=str,
        help="perceptions_1_to_6",
    )
    parser.add_argument(
        "--run_name",
        default="default",
        type=str,
        help="unique name to identify hyperparameter choices",
    )
    parser.add_argument("--data_dir", default="input/images/", type=str, help="dataset")
    return parser.parse_args()


def detect_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on %s device" % device)
    return device

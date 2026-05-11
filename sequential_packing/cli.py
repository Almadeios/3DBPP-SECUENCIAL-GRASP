import argparse


def buffer_value(text):
    value = int(text)
    if value < 1 or value > 10:
        raise argparse.ArgumentTypeError("buffer-size debe estar entre 1 y 10")
    return value


def positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("step debe ser positivo")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("este valor debe ser > 0")
    return value


def non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("este valor debe ser >= 0")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Empaque secuencial con caida y buffer.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Subdirectorio dentro de ./dataset (kitchen, blockout, ...).",
    )
    parser.add_argument(
        "--buffer-size",
        type=buffer_value,
        required=True,
        help="Tamano de la ventana K.",
    )
    parser.add_argument(
        "--step",
        type=positive_float,
        required=True,
        help="Resolucion del barrido (m).",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Secuencia a usar dentro de test_sequence.pt (default 0).",
    )
    parser.add_argument(
        "--restrict-rotations",
        action="store_true",
        help="Usa solo rotaciones ortogonales (desactiva poses estables de trimesh).",
    )
    parser.add_argument(
        "--tail-repack-size",
        type=non_negative_int,
        default=0,
        help="Tamano de bloque final a reordenar (0 desactiva la mejora).",
    )
    parser.add_argument(
        "--tail-repack-attempts",
        type=non_negative_int,
        default=1,
        help="Intentos deterministas para reinsertar la cola.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Semilla para reproducir comportamiento aleatorio.",
    )
    parser.add_argument(
        "--regular-order",
        "--zhao-order",
        dest="regular_order",
        action="store_true",
        help="Reordena la secuencia usando priorizando objetos voluminosos.",
    )
    parser.add_argument(
        "--irregular-order",
        action="store_true",
        help="Ordena priorizando formas irregulares (mejor para kitchen).",
    )
    parser.add_argument(
        "--max-passes",
        type=positive_int,
        default=1,
        help="Numero maximo de pasadas consecutivas reutilizando el estado.",
    )
    parser.add_argument(
        "--grasp-iterations",
        type=positive_int,
        default=1,
        help="Numero de construcciones GRASP (>=1).",
    )
    parser.add_argument(
        "--rcl-size",
        type=positive_int,
        default=3,
        help="Tamano de la lista restringida de candidatos.",
    )
    return parser.parse_args()

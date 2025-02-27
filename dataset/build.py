from dataset.imagenet import build_imagenet
from dataset.openimage import build_openimage
from dataset.t2i import build_laionas


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return 1000, build_imagenet(args, **kwargs), None
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'laionas':
        return build_laionas(args, **kwargs)

    
    raise ValueError(f'dataset {args.dataset} is not supported')
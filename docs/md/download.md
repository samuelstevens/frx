Module download
===============

Functions
---------

`download_url(url: str, path: str, chunk_size: int)`
:   

`main(args: download.Args)`
:   

Classes
-------

`Args(root: str = '.', imagenet_v2: bool = True, chunk_size_kb: int = 1)`
:   Args(root: str = '.', imagenet_v2: bool = True, chunk_size_kb: int = 1)

    ### Class variables

    `chunk_size_kb: int`
    :   how many KB to download at a time before writing to file.

    `imagenet_v2: bool`
    :   whether to download imagenet v2.

    `root: str`
    :   where to download files.
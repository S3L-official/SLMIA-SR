## step 1: check whether ffmpeg has been installed. 
- running `which ffmpeg` in your terminal
- if nothing outputs, go to step 2, since `ffmpeg` has not been installed.

- if it outputs an installed path, everything is ok.

## step 2: install ffmpeg
```
git clone https://git.ffmpeg.org/ffmpeg.git

cd ffmpeg

./configure --prefix=$HOME/.local/ffmpeg --disable-x86asm --enable-version3 --enable-shared

make; make install
```

## step 3: set the environment variable

- `vim ~/.bashrc`

- insert the following commands:
  ```
  export PATH=$HOME/.local/ffmpeg/bin:$PATH

  export LD_LIBRARY_PATH=$HOME/.local/ffmpeg/lib:$HOME/.local/lib:$LD_LIBRARY_PATH
  ```
  
- `source ~/.bashrc`
FROM archlinux
RUN mkdir worker
COPY ./* ./worker/

RUN sed -i '1s/^/Server = http:\/\/archlinux.cs.nctu.edu.tw\/$repo\/os\/$arch\n/' /etc/pacman.d/mirrorlist
RUN pacman -Sy base-devel --noconfirm
RUN pacman -Sy bazel cmake git tensorflow-cuda wget cuda cudnn nvidia hiredis clang --noconfirm

# RUN cd worker && make

# CMD ["worker/worker"]

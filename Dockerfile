FROM archlinux

RUN sed -i '1s/^/Server = http:\/\/archlinux.cs.nctu.edu.tw\/$repo\/os\/$arch\n/' /etc/pacman.d/mirrorlist
RUN pacman -Sy archlinux-keyring --noconfirm
RUN pacman -Sy base-devel --noconfirm
RUN pacman -Sy bazel cmake git tensorflow-cuda wget cuda cudnn nvidia hiredis clang --noconfirm

CMD ["/tf-worker/worker"]

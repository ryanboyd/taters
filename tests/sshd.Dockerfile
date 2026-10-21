# An ssh server for the tunnel tests: alpine, one user called taters who
# logs in with the public key given at build time, port forwarding allowed.
# See tests/README.md, "Live database tests".
FROM alpine:3.20
ARG PUBKEY
RUN apk add --no-cache openssh && ssh-keygen -A && adduser -D -s /bin/sh taters && passwd -u taters \
 && mkdir -p /home/taters/.ssh && chmod 700 /home/taters/.ssh \
 && echo "$PUBKEY" > /home/taters/.ssh/authorized_keys \
 && chown -R taters:taters /home/taters/.ssh && chmod 600 /home/taters/.ssh/authorized_keys \
 && sed -i 's/^#\?AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config \
 && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]

# An ssh server for the tunnel tests: alpine, one user called taters who
# logs in with the public key given at build time, port forwarding allowed.
# It also runs a MariaDB reachable only through its local socket, where the
# user taters logs in by socket authentication (no password) -- the shape
# of a lab server where `mysql` "just launches" once you are logged in.
# See tests/README.md, "Live database tests".
FROM alpine:3.20
ARG PUBKEY
RUN apk add --no-cache openssh mariadb mariadb-client && ssh-keygen -A \
 && adduser -D -s /bin/sh taters && passwd -u taters \
 && mkdir -p /home/taters/.ssh && chmod 700 /home/taters/.ssh \
 && echo "$PUBKEY" > /home/taters/.ssh/authorized_keys \
 && chown -R taters:taters /home/taters/.ssh && chmod 600 /home/taters/.ssh/authorized_keys \
 && sed -i 's/^#\?AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config \
 && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld \
 && mariadb-install-db --user=mysql --datadir=/var/lib/mysql >/dev/null \
 && (mariadbd --user=mysql --datadir=/var/lib/mysql --socket=/run/mysqld/mysqld.sock --skip-networking & \
     for i in $(seq 1 30); do mariadb --socket=/run/mysqld/mysqld.sock -e 'SELECT 1' >/dev/null 2>&1 && break; sleep 1; done; \
     mariadb --socket=/run/mysqld/mysqld.sock -e "CREATE DATABASE socket_test; CREATE USER 'taters'@'localhost' IDENTIFIED VIA unix_socket; GRANT ALL ON socket_test.* TO 'taters'@'localhost'; FLUSH PRIVILEGES;" \
     && mariadb-admin --socket=/run/mysqld/mysqld.sock shutdown)
EXPOSE 22
CMD sh -c "mariadbd --user=mysql --datadir=/var/lib/mysql --socket=/run/mysqld/mysqld.sock --skip-networking & exec /usr/sbin/sshd -D -e"

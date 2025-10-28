import logging
import time

from serial_interface import SerialInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    s = SerialInterface()

    # Use pyserial virtual loopback
    port = 'loop://'
    ok = s.connect(port, 115200)
    print('connect:', ok)

    # Write a line and read it back
    s.write_command('HELLO')
    time.sleep(0.05)
    line = s.read_line()
    print('read_line:', line)

    # Check get_available_ports still works (may be empty on CI)
    print('ports:', s.get_available_ports())


if __name__ == '__main__':
    main()

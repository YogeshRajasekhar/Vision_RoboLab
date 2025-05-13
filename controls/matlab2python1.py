import socket
import time

# Define server IP and port
server_ip = '127.0.0.1'
server_port = 5001

# Initialize the socket connection with a try-except
try:
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_client.connect((server_ip, server_port))
    print('Connected to the server.')
except Exception as e:
    raise Exception(f'Failed to connect to server. Check IP, port, and server status: {str(e)}')

# Data to send: eleven numbers
numbers_to_send = [-500, -1506, 1000, 0.1, 0.24, -0.9656539, 0, -1, 0, -2, 0]  # Replace with actual data
data_to_send = ';'.join([f'{num:.3f}' for num in numbers_to_send])  # Format as a semicolon-separated string

# Continuous communication loop
try:
    # Send the formatted string
    tcp_client.sendall(data_to_send.encode())
    print(f'Data sent to the server: {data_to_send}')
    
    # Wait for a response with a timeout
    time.sleep(0.5)  # Allow some time for the server to respond
    
    # Set non-blocking mode to check for data
    tcp_client.setblocking(False)
    
    try:
        # Try to read available data from the server
        response = tcp_client.recv(1024)
        print(f'Received from server: {response.decode()}')
    except BlockingIOError:
        print('No data available from the server.')
    
    # Set blocking mode back
    tcp_client.setblocking(True)
    
except Exception as e:
    # Handle errors gracefully
    print(f'Error during communication: {str(e)}')

finally:
    # Cleanup
    tcp_client.close()
    print('Connection closed.')
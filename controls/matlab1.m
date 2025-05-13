% Define server IP and port
serverIP = '127.0.0.1';
serverPort = 5001;

% Initialize the tcpclient object with a try-catch
try
    tcpClient = tcpclient(serverIP, serverPort);
    disp('Connected to the server.');
catch
    error('Failed to connect to server. Check IP, port, and server status.');
end

% Data to send: six numbers
numbersToSend = [-500;-1506;1000;0.1;0.24;-0.9656539;0;-1;0; -2;0]; % Replace with actual data
dataToSend = sprintf('%0.3f;%0.3f;%0.3f;%0.3f;%0.3f;%0.3f;%0.3f;%0.3f;%0.3f;%0.3f;%0.3f', numbersToSend); % Format as a comma-separated string with a newline

% Continuous communication loop
while isvalid(tcpClient)
    try
        % Send the formatted string
        write(tcpClient, dataToSend, 'string');
        disp(['Data sent to the server: ', dataToSend]);

        % Wait for a response with a timeout
        pause(0.5); % Allow some time for the server to respond

        if tcpClient.NumBytesAvailable > 0
            % Read available data from the server
            response = read(tcpClient, tcpClient.NumBytesAvailable, 'uint8');
            disp(['Received from server: ', char(response(:)')]);
        else
            disp('No data available from the server.');
        end

        break; % Exit the loop after sending once (remove for continuous communication)
    catch ME
        % Handle errors gracefully
        disp(['Error during communication: ', ME.message]);
        break; % Exit the loop if there is an issue
    end
end

% Cleanup
clear tcpClient;
disp('Connection closed.');
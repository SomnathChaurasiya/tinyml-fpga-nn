`timescale 1ns/1ps

module neuron(
    input clk,
    input rst,
    input start,
    input [9:0] N,

    input signed [7:0] in_data,
    input signed [7:0] weight,
    input signed [31:0] bias,

    output reg done,
    output reg signed [31:0] out,
    output reg [9:0] addr
);

    reg running;
    reg [9:0] count;
    reg signed [31:0] acc;

    always @(posedge clk) begin
        if (rst) begin
            running <= 0;
            count   <= 0;
            acc     <= 0;
            addr    <= 0;
            done    <= 0;
            out     <= 0;
        end else begin

            if (start && !running) begin
                running <= 1;
                count   <= 0;
                acc     <= 0;
                addr    <= 0;
                done    <= 0;
            end

            else if (running) begin
                acc <= acc + in_data * weight;

                if (count < N-1) begin
                    count <= count + 1;
                    addr  <= count + 1;
                end else begin
                    running <= 0;
                    done    <= 1;

                    if ((acc + bias) < 0)
                        out <= 0;
                    else
                        out <= acc + bias;
                end
            end

            else begin
                done <= 0;
            end
        end
    end

endmodule



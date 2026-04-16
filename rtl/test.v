`timescale 1ns/1ps

// ======================================================
// NEURON MODULE
// ======================================================
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


// ======================================================
// TESTBENCH WITH DEBUG PRINTS
// ======================================================
module neuron_tb;

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst, start;
    reg [9:0] N;

    reg signed [7:0] in_data, weight;
    reg signed [31:0] bias;

    wire done;
    wire signed [31:0] out;
    wire [9:0] addr;

    neuron uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .N(N),
        .in_data(in_data),
        .weight(weight),
        .bias(bias),
        .done(done),
        .out(out),
        .addr(addr)
    );

    // Memories
    reg signed [7:0] input_mem [0:783];
    reg signed [7:0] fc1_weight [0:50175];
    reg signed [31:0] fc1_bias [0:63];
    reg signed [31:0] layer1_out [0:63];

    reg signed [7:0] fc2_weight [0:639];
    reg signed [31:0] fc2_bias [0:9];
    reg signed [31:0] layer2_out [0:9];

    integer i, neuron_id, j;
    integer f1,f2,f3,f4,f5;

    integer max_idx;
    reg signed [31:0] max_val;

    reg layer_sel;

    initial begin
        rst = 1;
        start = 0;

        $display("\n================ TB START ================");

        // ---------------- FILE LOAD ----------------
        $display("\n[STEP 1] Loading files...");

        f1 = $fopen("/home/pavan/somnath_working_dir/tinyml_fpga/data/input.txt","r");
        f2 = $fopen("/home/pavan/somnath_working_dir/tinyml_fpga/data/fc1_weights.txt","r");
        f3 = $fopen("/home/pavan/somnath_working_dir/tinyml_fpga/data/fc1_bias.txt","r");
        f4 = $fopen("/home/pavan/somnath_working_dir/tinyml_fpga/data/fc2_weights.txt","r");
        f5 = $fopen("/home/pavan/somnath_working_dir/tinyml_fpga/data/fc2_bias.txt","r");

        if (f1==0 || f2==0 || f3==0 || f4==0 || f5==0) begin
            $display("❌ ERROR: File open failed");
            $finish;
        end

        for (i=0;i<784;i=i+1)
            $fscanf(f1,"%d\n",input_mem[i]);

        for (i=0;i<50176;i=i+1)
            $fscanf(f2,"%d\n",fc1_weight[i]);

        for (i=0;i<64;i=i+1)
            $fscanf(f3,"%d\n",fc1_bias[i]);

        for (i=0;i<640;i=i+1)
            $fscanf(f4,"%d\n",fc2_weight[i]);

        for (i=0;i<10;i=i+1)
            $fscanf(f5,"%d\n",fc2_bias[i]);

        $display("✅ Files loaded successfully");

        // Print sample values
        $display("\nSample INPUT:");
        for (i=0;i<5;i=i+1)
            $display("input[%0d] = %0d", i, input_mem[i]);

        $display("\nSample FC1 weights:");
        for (i=0;i<5;i=i+1)
            $display("w1[%0d] = %0d", i, fc1_weight[i]);

        #20 rst = 0;

        // ---------------- FC1 ----------------
        layer_sel = 0;
        N = 784;

        $display("\n[STEP 2] FC1 START (784→64)");

        for (neuron_id = 0; neuron_id < 64; neuron_id = neuron_id + 1) begin
            bias = fc1_bias[neuron_id];

            @(posedge clk); start = 1;
            @(posedge clk); start = 0;

            wait(done);
            @(posedge clk);

            layer1_out[neuron_id] = out;

            if (neuron_id < 5)
                $display("FC1[%0d] = %0d", neuron_id, out);

            wait(!done);
        end

        $display("✅ FC1 COMPLETED");

        // ---------------- FC2 ----------------
        layer_sel = 1;
        N = 64;

        $display("\n[STEP 3] FC2 START (64→10)");

        for (j = 0; j < 10; j = j + 1) begin
            bias = fc2_bias[j];

            @(posedge clk); start = 1;
            @(posedge clk); start = 0;

            wait(done);
            @(posedge clk);

            layer2_out[j] = out;

            $display("Class %0d = %0d", j, out);

            wait(!done);
        end

        $display("✅ FC2 COMPLETED");

        // ---------------- ARGMAX ----------------
        $display("\n[STEP 4] ARGMAX");

        max_val = layer2_out[0];
        max_idx = 0;

        for (i=1;i<10;i=i+1) begin
            if (layer2_out[i] > max_val) begin
                max_val = layer2_out[i];
                max_idx = i;
            end
        end

        $display("Max Score = %0d", max_val);
        $display("Predicted Digit = %0d", max_idx);

        // ---------------- FINAL ----------------
        $display("\n================ FINAL RESULT ================");
        $display("NN INFERENCE COMPLETE");
        $display("==============================================");

        #20 $finish;
    end

    // ---------------- DATA FEED ----------------
    always @(*) begin
        if (layer_sel == 0) begin
            in_data = input_mem[addr];
            weight  = fc1_weight[neuron_id*784 + addr];
        end else begin
            if (addr < 64) begin
                in_data = layer1_out[addr][7:0];
                weight  = fc2_weight[j*64 + addr];
            end else begin
                in_data = 0;
                weight  = 0;
            end
        end
    end

endmodule

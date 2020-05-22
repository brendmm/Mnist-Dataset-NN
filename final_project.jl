#############################################################################################
#               Final Project: MNIST Data Prediction                                        #
#                Author:        Brendan Muldowney                                           #
#                Version:       1                                                           #
#                Date:          12/11/19                                                    #
#                Description:   This script has three different neural networks implement   #
#                               to try and correctly classify the MNIST dataset, a          #
#                               collection of handwritten numbers seen as 28x28 pixles.     #
#                               This script will train and test in multiple layered networks#
#                               in order to allow the user to see which one is the best     #
#                                                                                           #
#                                                                                           #
#                *Note: Test and Train functions for each style neural network              #
#                       was designed and modified based of the instructor's, Christopher    #
#                       Wyatt's, Neural Network example code provided to the class          #
#############################################################################################

using Random, HDF5, LinearAlgebra, Distributions

#################################################
#               Activation Functions            #
#################################################
# sigmoid activation function
hsig(a) = 1 ./ (1 .+ exp.(-a))
# sigmoid derivative
hpsig(a) = hsig(a) .* (1 .- hsig(a))
# softMax activation function
softMax(input) = (exp.(input) ./ sum(exp.(input)));
# softMax derivative
softMaxDeriv(input) = (softMax(input)' .* (Matrix(I,length(input),length(input)) .- softMax(input)));

# activation function for output layer
hout = softMax
hpout = softMaxDeriv

# activation function for hidden layer
hhid = hsig
hphid = hpsig

#################################################################
#################################################################
#                                                               #
#                   1 Layer Neural Network                      #
#                                                               #
#################################################################
#################################################################

"This is the function to test our 1-layer network weights

    Inputs:
        weight1:    Matrix for layer 1
        input:      Enter input data as matrix
        target:     Enter target values as matrix
        idx:        Indexes for data to be tested
    Output:
        error:      Sum of error of all values tested
        miss/N:     Rate of missed calssifications
"
function test1Layer(weight1, input, target, idx)
    N = length(idx)

    M = size(weight1)[1]

    miss = 0
    error = 0
    for n = 1:N
        x = input[idx[n],:]
        t = target[idx[n],:]

        # forward propagate
        y = zeros(M)
        y = (weight1*x)
        z = softMax(y)
        if(argmax(z) != argmax(t))
            miss=miss+1
        end
        difference = (z.-t)
        error += dot(difference,difference)
    end
    println("Error: $(error)\tMissed $(miss) out of $(N)\tRate: $(miss/N*100)")

    return error, miss/N
end
function test1LayerTest(weight1, input, target, idx)
    N = length(idx)

    M = size(weight1)[1]

    miss = 0
    error = 0
    for n = 1:N
        x = input[idx[n],:]
        t = target[idx[n],:]

        # forward propagate
        y = zeros(M)
        y = (weight1*x)
        z = softMax(y)
        if(argmax(z) != argmax(t))
            miss=miss+1
        end
        difference = (z.-t)
        error += dot(difference,difference)
    end
    println("Error: $(error)\tMissed $(miss) out of $(N)\tRate: $(miss/N*100)")

    return error, miss/N
end
"This is the function to train our 1-layer network

    Inputs:
        input:      Enter input data as matrix
        target:     Enter target values as matrix
        speed:      Set to 1 to calcuate validation error only every 20 iterations
    Output:
        weight1:    Optimal matrix for layer 1
"
function train1Layer(input, target, speed)

    # number of samples
    N = size(target)[1]
    # dimension of input
    D = length(input[1,:])
    # number to hold out
    Nhold = round(Int64, N/3)
    # number in training set
    Ntrain = N - Nhold
    # create indices
    idx = shuffle(1:N)
    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]

    # number of nodes
    M = 10
    # batch size
    B = 500

    # layer 1 weights
    weight1 = 0.01*randn(M, D)
    bestweight1 = weight1

    pdf = Uniform(1,Ntrain)

    error = []

    stop = false

    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10^(-8)

    m1 = zeros(size(weight1))
    v1 = zeros(size(weight1))

    tau = 1

    println("Initial Validation Error:")
    bestError = test1Layer(weight1, input, target, testidx)[1]
    while !stop

        grad1 = zeros(M, D)

        for n = 1:B
            sample = trainidx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample,:]
            t = target[sample,:]
            # forward propagate
            y = zeros(M)
            y = (weight1*x)
            z = softMax(y)
            # end forward propagate
            # output error
            delta = (z.-t)

            grad1 += (delta*x')

        end

        m1 = beta1.*m1 .+ (1-beta1).*grad1

        v1 = beta2.*v1 .+ (1-beta2).*(grad1.^2)

        m1Hat = m1./(1-beta1^(tau))
        v1Hat = v1./(1-beta2^(tau))

        weight1 = weight1 .- (alpha.*m1Hat)./(sqrt.(v1Hat) .+ epsilon)

        tau = tau + 1
        if (tau%20 == 0 || speed != 1)
            print("$(tau):\t")
            temp = test1Layer(weight1, input, target, testidx)
            if(bestError>temp[1])
                bestError=temp[1]
                bestweight1=weight1
            end
            push!(error, temp[1])
            window = 10
            if length(error) > 2*window+1
                runningerror = error[(end-window):end]

                mean1 = mean(runningerror)
                mean2 = mean(error[(end-2*window):(end-window)])

                stop = mean1 > mean2
            end
        end
    end
    println("Final Training Error:")
    test1Layer(bestweight1, input, target, trainidx)
    println("Final Validation Error:")
    test1Layer(bestweight1, input, target, testidx)

    return bestweight1
end

#################################################################
#################################################################
#                                                               #
#                   2 Layer Neural Network                      #
#                                                               #
#################################################################
#################################################################

"This is the function to test our 2-layer network weights

    Inputs:
        weight1:    Matrix for layer 1
        weight2:    Matrix for layer 2
        input:      Enter input data as matrix
        target:     Enter target values as matrix
        idx:        Indexes for data to be tested
    Output:
        error:      Sum of error of all values tested
        miss/N:     Rate of missed calssifications
"
function test2Layer(weight1, weight2, input, target, idx)
    N = length(idx)

    M = size(weight1)[1]

    miss = 0
    error = 0
    for n = 1:N
        x = input[idx[n],:]
        t = target[idx[n],:]

        # forward propagate
        y = zeros(M+1)
        y[1] = 1 # bias node
        y[2:end] = hhid(weight1*x)
        a = (weight2*y)
        z = softMax(a)
        if(argmax(z) != argmax(t))
            miss=miss+1
        end
        difference = (z.-t)
        error += dot(difference,difference)
    end
    println("Error: $(error)\tMissed $(miss) out of $(N)\tRate: $(miss/N*100)")

    return error, miss/N
end
"This is the function to train our 2-layer network

    Inputs:
        input:      Enter input data as matrix
        target:     Enter target values as matrix
        speed:      Set to 1 to calcuate validation error only every 20 iterations
    Output:
        weight1:    Optimal matrix for layer 1
        weight2:    Optimal matrix for layer 2
"
function train2Layer(input, target, speed)

    # number of samples
    N = size(target)[1]
    # dimension of input
    D = length(input[1,:])
    # number to hold out
    Nhold = round(Int64, N/3)
    # number in training set
    Ntrain = N - Nhold
    # create indices
    idx = shuffle(1:N)
    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]

    # number of hidden nodes
    M = 1000

    # batch size
    B = 500

    T = 10

    # layer 1 weights
    weight1 = 0.01*randn(M, D)
    bestweight1 = weight1

    # layer 2 weights (inc bias)
    weight2 = 0.01*randn(T,M+1)
    bestweight2 = weight2

    numweights = prod(size(weight1)) + prod(size(weight2))
    # println("$(numweights) weights")

    pdf = Uniform(1,Ntrain)

    error = []

    stop = false

    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10^(-8)

    m1 = zeros(size(weight1))
    v1 = zeros(size(weight1))
    m2 = zeros(size(weight2))
    v2 = zeros(size(weight2))
    tau = 1
    println("Initial Validation Error:")
    bestError = test2Layer(weight1, weight2, input, target, testidx)[1]
    while !stop

        grad1 = zeros(M, D)
        grad2 = zeros(T,M+1)

        for n = 1:B
            sample = trainidx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample,:]
            t = target[sample,:]
            # forward propagate
            y = zeros(M+1)
            y[1] = 1 # bias node
            y[2:end] = hhid(weight1*x)

            output = weight2*y
            # println(size(output))
            # for i = 1:length(output)
            #     for j = 1:M+1
            #         output[i] += weight2[i,j]*y[j]
            #     end
            # end
            z = softMax(output)
            # end forward propagate
            # output error
            delta = (z.-t)

            grad2[:,1] += delta.*y[1]
            grad2[:,2:end] += (y[2:end]*delta'*hpout(output))'

            grad1 += ((weight2[:,2:end]'*delta).*hphid(weight1*x))*x'

        end
        grad1 = grad1./B
        grad2 = grad2./B

        m1 = beta1.*m1 .+ (1-beta1).*grad1
        v1 = beta2.*v1 .+ (1-beta2).*(grad1.^2)

        m2 = beta1.*m2 .+ (1-beta1).*grad2
        v2 = beta2.*v2 .+ (1-beta2).*(grad2.^2)

        m1Hat = m1./(1-beta1^(tau))
        v1Hat = v1./(1-beta2^(tau))

        m2Hat = m2./(1-beta1^(tau))
        v2Hat = v2./(1-beta2^(tau))
        weight1 = weight1 .- (alpha.*m1Hat)./(sqrt.(v1Hat) .+ epsilon)
        weight2 = weight2 .- (alpha.*m2Hat)./(sqrt.(v2Hat) .+ epsilon)

        tau = tau + 1
        if(tau%20 == 0 || speed != 1)
            print("$(tau):\t")
                temp = test2Layer(weight1, weight2, input, target, testidx)
                if(bestError>temp[1])
                    bestError=temp[1]
                    bestweight1=weight1
                    bestweight2=weight2
                end
                push!(error, temp[1])
                window = 5
                if length(error) > 2*window+1
                    runningerror = error[(end-window):end]

                    mean1 = mean(runningerror)
                    mean2 = mean(error[(end-2*window):(end-window)])

                    stop = mean1 > mean2
                end
        end

    end

    println("Final Training Error:")
    test2Layer(bestweight1, bestweight2, input, target, trainidx)
    println("Final Validation Error:")
    test2Layer(bestweight1, bestweight2, input, target, testidx)

    return bestweight1, bestweight2
end

#################################################################
#################################################################
#                                                               #
#                   3 Layer Neural Network                      #
#                                                               #
#################################################################
#################################################################

"This is the function to test our 3-layer network weights

    Inputs:
        weight1:    Matrix for layer 1
        weight2:    Matrix for layer 2
        weight3:    Matrix for layer 3
        input:      Enter input data as matrix
        target:     Enter target values as matrix
        idx:        Indexes for data to be tested
    Output:
        error:      Sum of error of all values tested
        miss/N:     Rate of missed calssifications
"
function test3Layer(weight1, weight2, weight3, input, target, idx)
    N = length(idx)

    M = size(weight1)[1]
    Q = size(weight2)[1]
    T = size(weight3)[1]

    miss = 0
    error = 0
    for n = 1:N
        x = input[idx[n],:]
        t = target[idx[n],:]

        # forward propagate
        y = zeros(M+1)
        y[1] = 1 # bias node
        y[2:end] = hhid(weight1*x)
        q = zeros(Q+1)
        q[1] = 1
        q[2:end] = hhid(weight2*y)
        a = (weight3*q)
        z = softMax(a)
        if(argmax(z) != argmax(t))
            miss=miss+1
        end
        difference = (z.-t)
        error += dot(difference,difference)
    end
    println("Error: $(error)\tMissed $(miss) out of $(N)\tRate: $(miss/N*100)")

    return error, miss/N
end
"This is the function to train our 3-layer network

    Inputs:
        input:      Enter input data as matrix
        target:     Enter target values as matrix
        speed:      Set to 1 to calcuate validation error only every 20 iterations
    Output:
        weight1:    Optimal matrix for layer 1
        weight2:    Optimal matrix for layer 2
        weight3:    Optimal matrix for layer 3
"
function train3Layer(input, target, speed)

    # number of samples
    N = size(target)[1]
    # dimension of input
    D = length(input[1,:])
    # number to hold out
    Nhold = round(Int64, N/3)
    # number in training set
    Ntrain = N - Nhold
    # create indices
    idx = shuffle(1:N)
    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]

    # number of hidden nodes in 1st Hidden Layer
    M = 700
    # number of hidden nodes in 2nd Hidden Layer
    Q = 300
    # batch size
    B = 500

    T = 10

    # layer 1 weights
    weight1 = 0.01*randn(M, D)
    bestweight1 = weight1

    # layer 2 weights (inc bias)
    weight2 = 0.01*randn(Q, M+1)
    bestweight2 = weight2

    # layer 3 weights (inc bias)
    weight3 = 0.01*randn(T,Q+1)
    bestweight3 = weight3

    pdf = Uniform(1,Ntrain)

    error = []

    stop = false

    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10^(-8)

    m1 = zeros(size(weight1))
    v1 = zeros(size(weight1))
    m2 = zeros(size(weight2))
    v2 = zeros(size(weight2))
    m3 = zeros(size(weight3))
    v3 = zeros(size(weight3))

    tau = 1
    println("Initial Validation Error:")
    bestError = test3Layer(weight1, weight2, weight3, input, target, testidx)[1]
    while !stop

        grad1 = zeros(M, D)
        grad2 = zeros(Q,M+1)
        grad3 = zeros(T,Q+1)

        for n = 1:B
            sample = trainidx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample,:]
            t = target[sample,:]
            # forward propagate
            y = zeros(M+1)
            y[1] = 1 # bias node
            y[2:end] = hhid(weight1*x)
            q = zeros(Q+1)
            q[1] = 1 # bias node
            q[2:end] = hhid(weight2*y)
            output = (weight3*q)

            z = softMax(output)
            # end forward propagate
            # output error
            delta = (z.-t)

            layer1Result = hphid(weight1*x)
            layer2Result = hphid(weight2*y)

            grad3[:,1] += delta.*q[1]
            grad3[:,2:end] += (q[2:end]*delta'*hpout(output))'

            tempError =  ((weight3[:,2:end]'*delta).*hphid(weight2*y))
            grad2 +=  tempError*y'

            grad1 += (tempError'*weight2[:,2:end])'.*hphid(weight1*x)*x'


        end
        grad1 = grad1./B
        grad2 = grad2./B
        grad3 = grad3./B

        m1 = beta1.*m1 .+ (1-beta1).*grad1
        v1 = beta2.*v1 .+ (1-beta2).*(grad1.^2)

        m2 = beta1.*m2 .+ (1-beta1).*grad2
        v2 = beta2.*v2 .+ (1-beta2).*(grad2.^2)

        m3 = beta1.*m3 .+ (1-beta1).*grad3
        v3 = beta2.*v3 .+ (1-beta2).*(grad3.^2)

        m1Hat = m1./(1-beta1^(tau))
        v1Hat = v1./(1-beta2^(tau))

        m2Hat = m2./(1-beta1^(tau))
        v2Hat = v2./(1-beta2^(tau))

        m3Hat = m3./(1-beta1^(tau))
        v3Hat = v3./(1-beta2^(tau))

        weight1 = weight1 .- (alpha.*m1Hat)./(sqrt.(v1Hat) .+ epsilon)
        weight2 = weight2 .- (alpha.*m2Hat)./(sqrt.(v2Hat) .+ epsilon)
        weight3 = weight3 .- (alpha.*m3Hat)./(sqrt.(v3Hat) .+ epsilon)

        tau = tau + 1
        if(tau%20 == 0 || speed != 1)
            print("$(tau):\t")
            temp = test3Layer(weight1, weight2, weight3, input, target, testidx)
            if(bestError>temp[1])
                bestError=temp[1]
                bestweight1=weight1
                bestweight2=weight2
                bestweight3=weight3
            end
            push!(error, temp[1])
            window = 5
            if length(error) > 2*window+1
                runningerror = error[(end-window):end]

                mean1 = mean(runningerror)
                mean2 = mean(error[(end-2*window):(end-window)])

                stop = mean1 > mean2
            end
        end
    end
    println("Final Training Error:")
    test3Layer(bestweight1, bestweight2, bestweight3, input, target, trainidx)
    println("Final Validation Error:")
    test3Layer(bestweight1, bestweight2, bestweight3, input, target, testidx)

    return bestweight1, bestweight2, bestweight3
end

#################################################################
#################################################################
#                                                               #
#                Main Function: Data Organization               #
#                                                               #
#################################################################
#################################################################
"This is the function to reorganize our images and labels

    Inputs:
        labels:    Enter labels
        images:    Enter images
    Output:
        data:       Image transformed into a vector
        target:     One hot encoded vector representing classification
"
function modifyData(labels,images)

    data = zeros(length(labels), 784)
    target = zeros(length(labels), 10)
    for i = 1:length(labels)
        data[i,1:784] = reshape(images[:,:,i], 1, 784)
        target[i,labels[i]+1] = 1
    end
    return data,target
end
"This is the function to run the 1-layer Neural Network

    Inputs:
        trainLabels:    Enter training labels
        trainImages:    Enter training images
        testLabels:     Enter testing labels
        testImages:     Enter testing images
        speed:          Set to 1 to calcuate validation error only every 20 iterations
    Output:
        None
"
function layer1(trainLabels,trainImages,testLabels,testImages,speed)
        data,target = modifyData(trainLabels,trainImages)
        practiceData = data[1:1000,:]
        practiceTarget = target[1:1000,:]
        w1 = train1Layer(data, target, speed)
        data,target = modifyData(testLabels,testImages)
        idx = shuffle(1:size(data)[1])
        println("---------------------------------------------------------------------------------------------------")
        println("---------------------------------------------------------------------------------------------------")
        println("Final Error:")
        testError = test1LayerTest(w1, data, target, idx)
        println("Finished")
end
"This is the function to run the 2-layer Neural Network

    Inputs:
        trainLabels:    Enter training labels
        trainImages:    Enter training images
        testLabels:     Enter testing labels
        testImages:     Enter testing images
        speed:          Set to 1 to calcuate validation error only every 20 iterations
    Output:
        None
"
function layer2(trainLabels,trainImages,testLabels,testImages,speed)
        data,target = modifyData(trainLabels,trainImages)
        practiceData = data[1:1000,:]
        practiceTarget = target[1:1000,:]
        values = train2Layer(data, target, speed)
        w1 = values[1]
        w2 = values[2]
        data,target = modifyData(testLabels,testImages)
        idx = shuffle(1:size(data)[1])
        println("---------------------------------------------------------------------------------------------------")
        println("---------------------------------------------------------------------------------------------------")
        println("Final Error:")
        testError = test2Layer(w1, w2, data, target, idx)
        println("Finished")
end
"This is the function to run the 3-layer Neural Network

    Inputs:
        trainLabels:    Enter training labels
        trainImages:    Enter training images
        testLabels:     Enter testing labels
        testImages:     Enter testing images
        speed:          Set to 1 to calcuate validation error only every 20 iterations
    Output:
        None
"
function layer3(trainLabels,trainImages,testLabels,testImages,speed)
    data,target = modifyData(trainLabels,trainImages)
    practiceData = data[1:1000,:]
    practiceTarget = target[1:1000,:]
    values = train3Layer(data, target, speed)
    w1 = values[1]
    w2 = values[2]
    w3 = values[3]
    data,target = modifyData(testLabels,testImages)
    idx = shuffle(1:size(data)[1])
    println("---------------------------------------------------------------------------------------------------")
    println("---------------------------------------------------------------------------------------------------")
    println("Final Error:")
    testError = test3Layer(w1, w2, w3, data, target, idx)
    println("Finished")
end

"This is the main function of the project. It controls which networks are run as
 well as how often the validation error is calculated.

    Inputs:
        run1:   Set to 1 to run the 1-layer neural network
        run2:   Set to 1 to run the 2-layer neural network
        run3:   Set to 1 to run the 3-layer neural network
        speed:  Set to 1 to calcuate validation error only every 20 iterations
    Output:
        None
"
function finalProject(run1,run2,run3,speed)
    h5open("mnist.h5", "r") do file

      trainLabels = read(file, "train/labels")
      trainImages = read(file, "train/images")
      testLabels = read(file, "test/labels")
      testImages = read(file, "test/images")

      if(run1 == 1)
          println("\n\n----------------------------------------------------------------------------------------------")
          println("\t\t\t1 Layer Neural Net")
          println("----------------------------------------------------------------------------------------------\n")
          @time layer1(trainLabels,trainImages,testLabels,testImages,speed);
      end
      if(run2 == 1)
          println("\n\n----------------------------------------------------------------------------------------------")
          println("\t\t\t2 Layer Neural Net")
          println("----------------------------------------------------------------------------------------------\n")
          @time layer2(trainLabels,trainImages,testLabels,testImages,speed);
      end
      if(run3 == 1)
          println("\n\n----------------------------------------------------------------------------------------------")
          println("\t\t\t3 Layer Neural Net")
          println("----------------------------------------------------------------------------------------------\n")
          @time layer3(trainLabels,trainImages,testLabels,testImages,speed);
      end
    end
end

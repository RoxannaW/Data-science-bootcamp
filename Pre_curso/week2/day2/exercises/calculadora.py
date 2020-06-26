  num_1 = int(input("Please type a number"))
    num_2 = int(input("please type a second number"))
    operador_1 = input("Please select a operador: Adición, Sustracción, Multiplicación o División")
    operador_2 = input("Please select a second operador: Adición, Sustracción, Multiplicación o División")

def calculadora(num1, num2, operador1, operador2):
    for elem in operador1:
        if elem == "adición":
            result = num1 + num2
            print("First result:" + str(result))
        elif elem == "Sustracción":
            result = num1 - num2
            print("First result:" + str(result))
        elif elem == "Multiplicación":
            result = num1 * num2
            print("First result:" + str(result))
        elif elem == "División":
            result = num1 / num2
            print("First result:" + str(result))
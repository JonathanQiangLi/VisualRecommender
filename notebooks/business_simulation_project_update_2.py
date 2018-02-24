import scipy.stats
import numpy as np
from random import randint
import pandas as pd
import math
import random
np.random.seed(seed=123)

class Business_simulation:


    def __init__(self,p,num_customer,session_range,num_product,a,b,p_0,p_1,p_2,beta,x0):
        self.p_gender = p
        self.num_customer = num_customer
        self.session_range = session_range
        self.num_product = num_product
        self.price_beta_a = a
        self.price_beta_b = b
        self.price_range_category_shoe = p_0
        self.price_range_category_cloth = p_1
        self.price_range_category_computer = p_2
        self.logit_function_scaling_factor = beta
        self.logit_function_scaling_factor = x0


    def customer_id_gender(self):
        """
            Part 0:
            create the customer information table: (customer_id_gender_df)
            (customer_id, gender):
        """
        customer_gender = scipy.stats.bernoulli.rvs(self.p_gender, size=self.num_customer)
        customer_id = list(range(self.num_customer))
        id_gender = np.column_stack((customer_id, customer_gender))
        customer_id_gender_df = pd.DataFrame(id_gender)
        customer_id_gender_df.columns = ["Customer_id", "Gender"]
        return customer_id_gender_df


    def simulate_customer_session(self, customer_id_gender_df):
        """
            Part 1:
            create the customer information table: (customer_session_df)
            Parameters: num_customer, p , session_range
            Customer ~ uniform(num_customer)
            Gender ~ bernoulli(p)
            session_length ~ uniform(session_range)
        """
        session_length = []
        for i in range(self.num_customer):
            x = np.random.randint(1, self.session_range+1)
            session_length.append(x)
        customer_id = []
        customer_gender = []
        for i in range(self.num_customer):
            x = np.random.randint(0, self.num_customer)
            y = customer_id_gender_df.loc[x][1]
            customer_id.append(x)
            customer_gender.append(y)
        customer_information = np.column_stack((customer_id, customer_gender, session_length))
        customer_session_df = pd.DataFrame(customer_information)
        customer_session_df.columns = ["Customer_id", "Gender", "session_length"]
        return customer_session_df


    def simulate_product_information(self):
        """
            Part 2:
            create product information table: (product_information_df)
            Parameters:
            num_product
            product_price ~ Beta(a,b)*(price for category 0,1,2: p_0,p_1,p_2)
        """
        category = []
        for i in range(self.num_product):
            x = np.random.randint(0,3)
            category.append(x)
        category_0 = category.count(0)
        category_1 = category.count(1)
        category_2 = category.count(2)
        list0 = list(np.zeros(category_0))
        list1 = list(np.zeros(category_1) + 1)
        list2 = list(np.zeros(category_2) + 2)
        category_list = list0 + list1 + list2
        product_0_price = scipy.stats.beta.rvs(self.price_beta_a, self.price_beta_b, size=category_0) * self.price_range_category_shoe
        product_1_price = scipy.stats.beta.rvs(self.price_beta_a, self.price_beta_b, size=category_1) * self.price_range_category_cloth
        product_2_price = scipy.stats.beta.rvs(self.price_beta_a, self.price_beta_b, size=category_2) * self.price_range_category_computer
        product_id = [int(i) for i in range(self.num_product)]
        prices = list(product_0_price) + list(product_1_price) + list(product_2_price)
        product_information = np.column_stack((product_id, category_list, prices))
        product_information_df = pd.DataFrame(product_information, dtype='float')
        product_information_df.columns = ["product_id", "category", "prices"]
        product_information_df["product_id"] = product_information_df["product_id"].astype("int")
        product_information_df["category"] = product_information_df["category"].astype("int")
        return product_information_df


    def simulate_shopping_information(self, customer_session_df):
        """
            Part 3:
            create shopping table for each customer: (shopping_df)
            which contains each product id for each customer at one session
            customer_session_df =
            simulate_customer_information(p,num_customer,session_range)
        """
        shopping_product_id = []
        for i in range(self.num_customer):
            list1 = []
            for item in range(customer_session_df['session_length'][i]):
                x = np.random.randint(0, self.num_product)
                list1.append(int(x))
            shopping_product_id.append(list1)
        shopping_product_id_df = pd.DataFrame(shopping_product_id, dtype='int')
        shopping_df = pd.concat([customer_session_df, shopping_product_id_df], axis=1)
        name = ["View_" + str(x) for x in range(100)]
        column_name = ["Customer_id", "Gender", "session_length"] + name
        shopping_df.columns = column_name
        shopping_df = shopping_df.fillna("")
        return shopping_df

    """
    Part 4: (total_0_1_df)
    for the each session: calculate the product is brought or not?(1: Brought, 0: not brought)
    p(buy|product,customer) = p(buy|customer_gender,product_category, product_price)
    log(p/(1-p))=alpha_0 + alpha_11*Indicator{male}*{category_1} +
      alpha_12*Indicator{male}*{category_2}+alpha_13*Indicator{male}*{category_3}
      +[alpha_21*price+alpha_31*price^2+alpha_41]*{category_1}
      +[alpha_22*price+alpha_32*price^2+alpha_42]*{category_2}
      +[alpha_23*price+alpha_33*price^2+alpha_43]*{category_3}  
      alpha_0 = log(0.1/(1-0.1))
      alpha_11 = 1; alpha_12 = -1; alpha_13 = -1
    """

    # define logistic regression model:
    # the parameter : x which is the price
    # lr_m0,lr_m1,lr_m2: for male(0) and product category 0,1,2
    # lr_f0,lr_f1,lr_f2: for female(1) and product category 0,1,2
    @staticmethod
    def lr_m0(x):
        return math.log(1.0 / 9) - 1 + (-10.0 / (100 ** 2) * (x - 100) ** 2 + 10.0)

    @staticmethod
    def lr_m1(x):
        return math.log(1.0 / 9) - 1 + (-10.0 / (200 ** 2) * (x - 200) ** 2 + 10.0)

    @staticmethod
    def lr_m2(x):
        return math.log(1.0 / 9) + 1 + (-1.0 / (4000 ** 2) * (x - 4000) ** 2 + 1.0)

    @staticmethod
    def lr_f0(x):
        return math.log(1.0 / 9) + 1 + (-10.0 / (100 ** 2) * (x - 100) ** 2 + 10.0)

    @staticmethod
    def lr_f1(x):
        return math.log(1.0 / 9) + 1 + (-10.0 / (200 ** 2) * (x - 200) ** 2 + 10.0)

    @staticmethod
    def lr_f2(x):
        return math.log(1.0 / 9) - 1 + (-1.0 / (4000 ** 2) * (x - 4000) ** 2 + 1.0)


    def each_customer_buy_0_1(self, buying_id, shopping_df, product_information_df):
        """
            calculate the probability for each view(product) at one session: log(p/(1-p))=(logit-x0)*beta
            shopping_df = simulate_shopping_information(num_customer,num_product,customer_session_df)
            product_information_df = simulate_product_information(num_product,a,b,p_0,p_1,p_2)
            determine the product is brought or not! P>=0.5 brought.
        """
        buying_or_not = []
        for i in range(int(shopping_df.loc[buying_id][2])):
            gender = shopping_df.loc[buying_id][1]
            x = shopping_df.loc[buying_id][3 + int(i)]
            category = product_information_df.loc[int(x)]['category']
            price = product_information_df.loc[int(x)]['prices']
            if gender == 0 and category == 0:
                logit = self.lr_m0(price)
            elif gender == 0 and category == 1:
                logit = self.lr_m1(price)
            elif gender == 0 and category == 2:
                logit = self.lr_m2(price)
            elif gender == 1 and category == 0:
                logit = self.lr_f0(price)
            elif gender == 1 and category == 1:
                logit = self.lr_f1(price)
            elif gender == 1 and category == 2:
                logit = self.lr_f2(price)
            else:
                logit = math.log(1.0 / 9)
            P_buy = 1 / (1 + math.exp(-(logit - self.logit_function_scaling_factor) * self.logit_function_scaling_factor))
            if P_buy >= 0.5:
                buying_or_not.append(1)
            else:
                buying_or_not.append(0)
        return buying_or_not


    def simulate_total_buying01(self, shopping_df, product_information_df):
        """
            Part 5: (simulate_total_buying01)
            for all customers calculate the number of product brought at one session
        """
        total_buying_0_1 = []
        for i in range(self.num_customer):
            x = self.each_customer_buy_0_1(i, shopping_df, product_information_df)
            total_buying_0_1.append(x)
        total_0_1_df = pd.DataFrame(total_buying_0_1)
        name = ["View_" + str(x) for x in range(100)]
        total_0_1_df.columns = name
        total_0_1_df = total_0_1_df.fillna("")
        return total_0_1_df


    def simulate_whole_data(self, simulated_buying_0_1, shopping_df):
        """
            Part 6:(whole_data)
            Data transformation:
            for each row: transform each session(viewed product) to rows
        """
        whole_data = pd.DataFrame()
        for item in list(range(0, self.num_customer)):
            id = item
            customer_list = []
            gender_list = []
            session_length_list = []
            # change the gender 0-M; 1-F
            if simulated_buying_0_1['Gender'][id] == 0:
                for i in range(simulated_buying_0_1['session_length'][id]):
                    gender_list.append("M")
            else:
                for i in range(simulated_buying_0_1['session_length'][id]):
                    gender_list.append("F")
            # get the product id
            for i in range(simulated_buying_0_1['session_length'][id]):
                customer_list.append(int(shopping_df.loc[id][0]))
            # get the orginal index
            for i in range(simulated_buying_0_1['session_length'][id]):
                session_length_list.append(id)
            buy01 = simulated_buying_0_1.loc[id][3:int(3 + simulated_buying_0_1['session_length'][id])].values
            product_id_i = shopping_df.loc[id][3:int(3 + simulated_buying_0_1['session_length'][id])].values
            df_i = pd.concat([pd.DataFrame(customer_list), pd.DataFrame(gender_list), pd.DataFrame(product_id_i),pd.DataFrame(buy01), pd.DataFrame(session_length_list)], axis=1)
            whole_data = pd.concat([whole_data, df_i])
        whole_data.columns = ["Customer_id", "Gender", "Product_id", "Brought", "index"]
        whole_data = pd.DataFrame(whole_data)
        whole_data['Customer_id'] = whole_data['Customer_id'].astype(int)
        whole_data['Gender'] = whole_data['Gender'].astype(str)
        whole_data['Product_id'] = whole_data['Product_id'].astype(float).astype(int)
        whole_data['Brought'] = whole_data['Brought'].astype(float).astype(int)
        whole_data['index'] = whole_data['index'].astype(int)
        whole_data.reset_index(drop=True, inplace=True)
        whole_data["Num_session"] = whole_data.groupby(["Customer_id"])["index"].rank(method="dense", ascending=True)
        whole_data["Num_session"] = whole_data["Num_session"].astype(int)
        whole_data.drop('index', axis=1, inplace=True)
        return whole_data



def main():

    simulation_data = Business_simulation(p=0.5,num_customer=1000,session_range=100,
                                          a=2,b=5,p_0=200,p_1=400,p_2=8000,
                                          num_product=1000,beta=0.1,x0=8)


    # part0:
    customer_id_gender_df = simulation_data.customer_id_gender()
    print("Customer information table =====> ")
    print(customer_id_gender_df.loc[0:5])

    # part1:
    customer_session_df = simulation_data.simulate_customer_session(customer_id_gender_df)
    print("Session information table ====> ")
    print(customer_session_df[0:5])

    # part2:
    product_information_df = simulation_data.simulate_product_information()
    print("Product information table ====> ")
    print(product_information_df[0:5])

    # part3:
    shopping_df = simulation_data.simulate_shopping_information(customer_session_df)
    print("Shopping information (product id) ====> ")
    print(shopping_df[0:5])

    # part4,5:
    total_0_1_df = simulation_data.simulate_total_buying01(shopping_df, product_information_df)
    simulated_buying_0_1 = pd.concat([customer_session_df, total_0_1_df], axis=1)
    print("Shopping information (which product is bought) ====>")
    print(simulated_buying_0_1[0:5])

    # part6:
    whole_data = simulation_data.simulate_whole_data(simulated_buying_0_1, shopping_df)
    print("Total information table (for each customer in one session) ====>")
    print(whole_data.sort_values(by=['Num_session'], ascending=False))
    #print(pd.value_counts(shopping_df['Customer_id'])[0:5])

    #whole_data.to_csv('/Users/dan/Desktop/whole_data.csv')


if __name__== "__main__":
  main()





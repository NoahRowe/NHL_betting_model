import tensorflow as tf

def calc_r(A, P, t):
    return (A/P)**(1/t) - 1


def get_fractional_loss(batch_size, day_bet_num):
    BANKROLL_START, BANKRUPT_VAL, BS = 100., 100., float(batch_size)
    
    def fractional_loss(y_true, y_pred):
        # Define home bets as above 0, away bets as below 0
        bet_values = tf.abs(y_pred)
        bet_home = tf.divide(tf.add(tf.divide(y_pred, bet_values+0.0001), 1), 2)
        bet_away = tf.multiply(tf.subtract(bet_home, 1), -1)
        
        home_bet = tf.multiply(bet_values, bet_home)
        away_bet = tf.multiply(bet_values, bet_away)

        home_won = tf.reshape(tf.slice(y_true, [0,0], [-1,1]), (-1,1))
        away_won = tf.reshape(tf.add(tf.multiply(home_won, -1), +1), (-1,1))

        home_odds = tf.add(tf.reshape(tf.slice(y_true, [0,1], [-1,1]), (-1,1)), -1)
        away_odds = tf.add(tf.reshape(tf.slice(y_true, [0,2], [-1,1]), (-1,1)), -1)

        # Get gain from winning
        a1 = tf.multiply(home_won, home_bet) # amount bet on home winners
        b1 = tf.multiply(away_won, away_bet) # amount bet on away winners
        t1 = tf.add(a1, b1) # amount bet on winners

        a2 = tf.multiply(home_won, home_odds) # odds on home winners
        b2 = tf.multiply(away_won, away_odds) # odds on away winners
        t2 = tf.add(a2, b2) # odds on winners

        gain = tf.multiply(t1, t2) # odds on winners * ammount bet on winners 

        # Get loss from losing
        a = tf.multiply(home_won, away_bet)   # amount bet on away losers
        b = tf.multiply(away_won, home_bet)   # amount bet on home losers
        loss = tf.multiply(tf.add(a, b), -1)  # amount bet on losers, negative

        # Sum them together
        retVal = tf.math.add(gain, loss)      # gains plus losses
        retVal = tf.reshape(retVal, (-1, 1))
        
        bankroll = BANKROLL_START
        day_roll = bankroll
        for i in range(batch_size):
            bankroll += day_roll*retVal[i]
            if i%day_bet_num==0:
                day_roll = bankroll

        bankroll_tensor = tf.convert_to_tensor([bankroll])
        bankroll = tf.where(bankroll_tensor<tf.constant(0.0), tf.constant(0.01), bankroll_tensor)

        interest_rate = tf.math.pow(tf.divide(tf.nn.relu(bankroll), BANKROLL_START), tf.divide(1., BS)) - 1.
        
        return -1 * interest_rate
    
    return fractional_loss


def get_unit_bet_loss(batch_size, day_bet_num):
    BANKROLL_START, BANKRUPT_VAL, BS = 100., 100., float(batch_size)
    UNIT_SIZE = 10
    
    def unit_bet_loss(y_true, y_pred):
        # Define home bets as above 0, away bets as below 0
        
#         FOR DOUBLE OUTPUT
#         [bet home, bet away]
#         home_preds = tf.reshape(tf.slice(y_pred, [0,0], [-1,1]), (-1,1))
#         away_preds = tf.reshape(tf.slice(y_pred, [0,1], [-1,1]), (-1,1))
        
#         # Find what value is higher - home or away preds
#         difference = tf.subtract(home_preds, away_preds)
#         bet_home = tf.divide(tf.add(tf.divide(difference, tf.abs(difference)), 1), 2)
#         bet_away = tf.multiply(tf.subtract(bet_home, 1), -1)
        
        # FOR SINGLE OUTPUT WITH SHOULD_BET
#         [bet_home, should_bet]
        should_bet = tf.subtract(tf.reshape(tf.slice(y_pred, [0,1], [-1,1]), (-1,1)), 0.4)
        should_bet_abs = tf.abs(should_bet)
        should_bet = tf.divide(tf.add(tf.divide(should_bet, should_bet_abs+0.0001), 1), 2)
        
        y_pred = tf.reshape(tf.slice(y_pred, [0,0], [-1,1]), (-1,1))
        y_pred = tf.multiply(should_bet, y_pred)
        
        y_pred = tf.subtract(tf.reshape(y_pred, (-1,1)), 0.5)
        bet_values = tf.abs(y_pred)
        bet_home = tf.divide(tf.add(tf.divide(y_pred, bet_values+0.0001), 1), 2)
        bet_away = tf.multiply(tf.subtract(bet_home, 1), -1)

        home_won = tf.reshape(tf.slice(y_true, [0,0], [-1,1]), (-1,1))
        away_won = tf.reshape(tf.add(tf.multiply(home_won, -1), +1), (-1,1))

        home_odds = tf.add(tf.reshape(tf.slice(y_true, [0,1], [-1,1]), (-1,1)), -1)
        away_odds = tf.add(tf.reshape(tf.slice(y_true, [0,2], [-1,1]), (-1,1)), -1)

        # Get gain from winning
        a1 = tf.multiply(home_won, bet_home) # amount bet on home winners
        b1 = tf.multiply(away_won, bet_away) # amount bet on away winners
        t1 = tf.add(a1, b1)                  # amount bet on winners

        a2 = tf.multiply(home_won, home_odds) # odds on home winners
        b2 = tf.multiply(away_won, away_odds) # odds on away winners
        t2 = tf.add(a2, b2)                   # odds on winners

        gain = tf.multiply(t1, t2) # odds on winners * ammount bet on winners 

        # Get loss from losing
        a = tf.multiply(home_won, bet_away)   # amount bet on away losers
        b = tf.multiply(away_won, bet_home)   # amount bet on home losers
        loss = tf.multiply(tf.add(a, b), -1)  # amount bet on losers, negative

        # Sum them together
        retVal = tf.math.add(gain, loss)      # gains plus losses
        retVal = tf.reshape(retVal, (-1, 1))
        
        bankroll = BANKROLL_START
        for i in range(batch_size):
            bankroll += UNIT_SIZE*retVal[i]

        return -bankroll
#         bankroll_tensor = tf.convert_to_tensor([bankroll])
#         bankroll = tf.where(bankroll_tensor<tf.constant(0.0), tf.constant(0.01), bankroll_tensor)

#         interest_rate = tf.math.pow(tf.divide(tf.nn.relu(bankroll), BANKROLL_START), tf.divide(1., BS)) - 1.
        
#         return -1 * interest_rate
    
    return unit_bet_loss
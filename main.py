import Reinforcement.Categorical.CategoricalUI as CUI
import Reinforcement.Categorical.CategoricalTestEnv as TEnv

UI = CUI.PolicyGradientUI(TEnv.TestEnv())
UI.mainloop()
import numpy as np
from .objective_functions import load_sample_data

class MultiObjectiveGA:
    def __init__(self, pop_size=100, n_generations=100, mutation_rate=0.1, data_file=None):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.bounds = [(-5, 5)] * 6  # 6次元の探索範囲
        self.data_file = data_file
        
        # データファイルが指定されている場合は読み込む
        if self.data_file:
            self.X, self.f1, self.f2 = load_sample_data(self.data_file)
            print(f"データファイル {self.data_file} から {len(self.X)} 件のデータを読み込みました。")
        else:
            raise ValueError("データファイルが指定されていません。--data-fileオプションで指定してください。")
        
    def initialize_population(self):
        """初期集団の生成"""
        # データファイルから読み込んだデータを使用
        indices = np.random.choice(len(self.X), self.pop_size, replace=True)
        return self.X[indices]
    
    def evaluate(self, individual):
        """個体の評価"""
        # 最も近いデータポイントを見つける
        distances = np.sum((self.X - individual)**2, axis=1)
        closest_idx = np.argmin(distances)
        return self.f1[closest_idx], self.f2[closest_idx]
    
    def dominates(self, obj1, obj2):
        """パレート支配関係の判定"""
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    def non_dominated_sort(self, population):
        """非支配ソート"""
        n = len(population)
        domination_sets = [[] for _ in range(n)]
        domination_counts = np.zeros(n)
        
        # 支配関係の計算
        for i in range(n):
            for j in range(i + 1, n):
                obj_i = self.evaluate(population[i])
                obj_j = self.evaluate(population[j])
                
                if self.dominates(obj_i, obj_j):
                    domination_sets[i].append(j)
                    domination_counts[j] += 1
                elif self.dominates(obj_j, obj_i):
                    domination_sets[j].append(i)
                    domination_counts[i] += 1
        
        # フロントの作成
        fronts = [[]]
        for i in range(n):
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in domination_sets[j]:
                    domination_counts[k] -= 1
                    if domination_counts[k] == 0:
                        next_front.append(k)
            i += 1
            fronts.append(next_front)
        
        return fronts
    
    def crossover(self, parent1, parent2):
        """交叉"""
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    
    def mutate(self, individual):
        """突然変異"""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                individual[i] = np.random.uniform(low, high)
        return individual
    
    def optimize(self):
        """最適化の実行"""
        population = self.initialize_population()
        pareto_front = []
        
        for generation in range(self.n_generations):
            # 評価と非支配ソート
            fronts = self.non_dominated_sort(population)
            
            # 新しい個体の生成
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(population[front])
                else:
                    remaining = self.pop_size - len(new_population)
                    new_population.extend(population[front[:remaining]])
                    break
            
            # パレート最適解の保存
            pareto_front = population[fronts[0]]
            
            # 次世代の生成
            while len(new_population) < self.pop_size:
                parent1 = population[np.random.randint(len(population))]
                parent2 = population[np.random.randint(len(population))]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = np.array(new_population)
        
        return pareto_front 
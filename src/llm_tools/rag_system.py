"""
RAG (Retrieval-Augmented Generation) System for MOF Research

检索增强生成系统，结合向量数据库和LLM提供准确的MOF知识查询

功能：
- 文档向量化和存储
- 语义相似度检索
- 上下文增强的LLM生成
- 知识来源追踪
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from collections import defaultdict


class SimpleVectorStore:
    """
    简化的向量数据库

    实际应用中应使用：
    - FAISS (Facebook AI Similarity Search)
    - Chroma
    - Pinecone
    - Weaviate
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ):
        """添加文档到向量库"""
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float, Dict]]:
        """
        语义相似度搜索

        返回: [(document, similarity_score, metadata), ...]
        """
        if len(self.embeddings) == 0:
            return []

        # 计算余弦相似度
        embeddings_matrix = np.array(self.embeddings)

        # 归一化
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        embeddings_norm = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10)

        # 余弦相似度
        similarities = embeddings_norm @ query_norm

        # 获取top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((
                self.documents[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))

        return results


class SimpleEmbedder:
    """
    简化的文本嵌入器

    实际应用中应使用：
    - sentence-transformers (HuggingFace)
    - OpenAI embeddings
    - Cohere embeddings
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

        # 简化的词汇表（实际应使用预训练模型）
        self.vocab = {}
        self.vocab_size = 1000

    def embed(self, text: str) -> np.ndarray:
        """
        文本嵌入（简化版）

        实际应该使用预训练模型
        """
        # 简化：基于文本特征的伪嵌入
        text_lower = text.lower()

        # 创建基于关键词的特征向量
        embedding = np.random.randn(self.embedding_dim) * 0.1

        # MOF相关关键词加权
        mof_keywords = {
            'mof': 1.0, 'uio': 0.8, 'hkust': 0.8, 'zif': 0.8,
            'zr': 0.6, 'cu': 0.6, 'zn': 0.6,
            'bdc': 0.7, 'btc': 0.7,
            'co2': 0.9, 'uptake': 0.8, 'adsorption': 0.8,
            'surface area': 0.9, 'stability': 0.7
        }

        for keyword, weight in mof_keywords.items():
            if keyword in text_lower:
                # 为包含关键词的文本添加特定模式
                idx = hash(keyword) % self.embedding_dim
                embedding[idx] += weight

        # 归一化
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        embeddings = [self.embed(text) for text in texts]
        return np.array(embeddings)


class MOF_RAG_System:
    """
    MOF检索增强生成系统

    完整的RAG流程：
    1. 文档分块和嵌入
    2. 向量存储
    3. 查询检索
    4. 上下文增强的LLM生成
    """

    def __init__(
        self,
        llm_model=None,
        embedder=None,
        vector_store=None
    ):
        self.llm = llm_model
        self.embedder = embedder if embedder else SimpleEmbedder()
        self.vector_store = vector_store if vector_store else SimpleVectorStore()

        self.documents_added = 0

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        chunk_size: int = 500
    ):
        """
        添加文档到知识库

        参数:
            documents: 文档列表（如论文文本）
            metadata: 元数据（如作者、年份、MOF名称）
            chunk_size: 分块大小
        """
        print(f"Adding {len(documents)} documents to knowledge base...")

        # 文档分块
        chunks = []
        chunk_metadata = []

        for i, doc in enumerate(documents):
            doc_chunks = self._chunk_text(doc, chunk_size)
            chunks.extend(doc_chunks)

            # 为每个chunk添加元数据
            doc_meta = metadata[i] if metadata and i < len(metadata) else {}
            for j, chunk in enumerate(doc_chunks):
                chunk_meta = doc_meta.copy()
                chunk_meta['chunk_id'] = j
                chunk_meta['doc_id'] = i
                chunk_metadata.append(chunk_meta)

        # 生成嵌入
        embeddings = self.embedder.embed_batch(chunks)

        # 存储到向量库
        self.vector_store.add_documents(chunks, embeddings, chunk_metadata)

        self.documents_added += len(documents)
        print(f"✓ Added {len(chunks)} chunks from {len(documents)} documents")
        print(f"  Total documents in knowledge base: {self.documents_added}")

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """将文本分块"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def query(
        self,
        question: str,
        top_k: int = 3,
        return_sources: bool = True
    ) -> Dict:
        """
        查询知识库并生成回答

        参数:
            question: 用户问题
            top_k: 检索的文档数量
            return_sources: 是否返回来源

        返回:
            result: {
                'answer': str,
                'sources': List[Dict],
                'context_used': str
            }
        """
        # 1. 嵌入查询
        query_embedding = self.embedder.embed(question)

        # 2. 检索相关文档
        retrieved_docs = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k
        )

        if not retrieved_docs:
            return {
                'answer': "I don't have enough information in my knowledge base to answer this question.",
                'sources': [],
                'context_used': ''
            }

        # 3. 构建上下文
        context = self._build_context(retrieved_docs)

        # 4. 生成回答
        answer = self._generate_answer(question, context)

        # 5. 整理来源信息
        sources = []
        if return_sources:
            for doc, score, meta in retrieved_docs:
                sources.append({
                    'text': doc[:200] + "..." if len(doc) > 200 else doc,
                    'similarity': score,
                    'metadata': meta
                })

        return {
            'answer': answer,
            'sources': sources,
            'context_used': context
        }

    def _build_context(self, retrieved_docs: List[Tuple]) -> str:
        """从检索到的文档构建上下文"""
        context_parts = []

        for i, (doc, score, meta) in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.3f})")
            if meta:
                context_parts.append(f"Metadata: {json.dumps(meta)}")
            context_parts.append(doc)
            context_parts.append("")

        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """使用LLM生成回答"""
        if self.llm is None:
            # 简化的回答生成（无LLM）
            return self._simple_answer_generation(question, context)

        # 使用真实LLM
        prompt = f"""
Based on the following context from MOF research literature, please answer the question.

Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information, say so.

Answer:
"""

        return self.llm.generate(prompt)

    def _simple_answer_generation(self, question: str, context: str) -> str:
        """简化的回答生成（无需LLM）"""
        # 提取关键信息
        lines = context.split('\n')
        relevant_lines = [l for l in lines if any(
            kw in l.lower() for kw in question.lower().split()
        )]

        if not relevant_lines:
            return "Based on the retrieved documents: " + lines[0][:200]

        return "Based on the retrieved documents: " + " ".join(relevant_lines[:3])[:500]


class KnowledgeBase:
    """
    MOF知识库管理器

    管理和组织MOF相关的文档、数据和知识
    """

    def __init__(self):
        self.papers = []
        self.mof_database = {}
        self.synthesis_protocols = {}

    def add_paper(
        self,
        title: str,
        authors: str,
        year: int,
        abstract: str,
        full_text: Optional[str] = None
    ):
        """添加论文到知识库"""
        paper = {
            'title': title,
            'authors': authors,
            'year': year,
            'abstract': abstract,
            'full_text': full_text or abstract
        }
        self.papers.append(paper)

    def add_mof_entry(
        self,
        mof_name: str,
        metal: str,
        linker: str,
        properties: Dict,
        synthesis: Optional[Dict] = None
    ):
        """添加MOF条目"""
        self.mof_database[mof_name] = {
            'metal': metal,
            'linker': linker,
            'properties': properties,
            'synthesis': synthesis or {}
        }

    def get_all_documents(self) -> List[str]:
        """获取所有文档（用于RAG系统）"""
        documents = []

        # 添加论文
        for paper in self.papers:
            doc = f"Title: {paper['title']}\n"
            doc += f"Authors: {paper['authors']} ({paper['year']})\n"
            doc += f"Abstract: {paper['abstract']}\n"
            if paper['full_text'] != paper['abstract']:
                doc += f"Content: {paper['full_text']}"
            documents.append(doc)

        # 添加MOF数据库条目
        for mof_name, data in self.mof_database.items():
            doc = f"MOF: {mof_name}\n"
            doc += f"Metal: {data['metal']}\n"
            doc += f"Linker: {data['linker']}\n"
            doc += f"Properties: {json.dumps(data['properties'])}\n"
            if data['synthesis']:
                doc += f"Synthesis: {json.dumps(data['synthesis'])}"
            documents.append(doc)

        return documents

    def get_metadata(self) -> List[Dict]:
        """获取所有文档的元数据"""
        metadata = []

        # 论文元数据
        for paper in self.papers:
            metadata.append({
                'type': 'paper',
                'title': paper['title'],
                'year': paper['year']
            })

        # MOF条目元数据
        for mof_name in self.mof_database:
            metadata.append({
                'type': 'mof_entry',
                'mof_name': mof_name
            })

        return metadata


# ===== 使用示例 =====

def example_rag_system():
    """完整的RAG系统示例"""
    print("=" * 70)
    print("RAG System for MOF Research")
    print("=" * 70)

    # 1. 创建知识库
    print("\n[Step 1] Building Knowledge Base...")
    kb = KnowledgeBase()

    # 添加论文
    kb.add_paper(
        title="UiO-66: A Highly Stable Zr-MOF",
        authors="Cavka et al.",
        year=2008,
        abstract="UiO-66 is a zirconium-based MOF with exceptional water and thermal stability. "
                 "It is synthesized from ZrCl4 and terephthalic acid (H2BDC) in DMF at 120°C. "
                 "The framework exhibits a BET surface area of 1200 m²/g and shows promise for "
                 "CO2 capture applications with an uptake of 3.0 mmol/g at 298K."
    )

    kb.add_paper(
        title="HKUST-1: A Copper-Based MOF with Open Metal Sites",
        authors="Chui et al.",
        year=1999,
        abstract="HKUST-1 (also Cu-BTC) features copper paddle-wheel units connected by "
                 "1,3,5-benzenetricarboxylate linkers. The material shows high CO2 uptake "
                 "(6.5 mmol/g) due to exposed copper sites. However, it is sensitive to moisture. "
                 "Synthesis involves Cu(NO3)2 and H3BTC in ethanol/water at 120°C."
    )

    # 添加MOF数据库条目
    kb.add_mof_entry(
        mof_name="MOF-5",
        metal="Zn",
        linker="BDC",
        properties={'surface_area': 3800, 'co2_uptake': 4.5},
        synthesis={'solvent': 'DEF', 'temperature': '100°C'}
    )

    print(f"✓ Knowledge base contains {len(kb.papers)} papers and {len(kb.mof_database)} MOF entries")

    # 2. 创建RAG系统
    print("\n[Step 2] Creating RAG System...")
    rag = MOF_RAG_System()

    # 3. 添加文档到RAG系统
    print("\n[Step 3] Indexing Documents...")
    documents = kb.get_all_documents()
    metadata = kb.get_metadata()

    rag.add_documents(documents, metadata, chunk_size=300)

    # 4. 查询测试
    print("\n[Step 4] Testing Queries...")

    questions = [
        "What is the CO2 uptake of UiO-66?",
        "Which MOF has open metal sites?",
        "How is HKUST-1 synthesized?",
        "What is the surface area of MOF-5?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {question}")
        print('-'*70)

        result = rag.query(question, top_k=2)

        print(f"\n📝 Answer:")
        print(f"  {result['answer'][:300]}...")

        print(f"\n📚 Sources ({len(result['sources'])}):")
        for j, source in enumerate(result['sources'], 1):
            print(f"\n  Source {j} (Similarity: {source['similarity']:.3f}):")
            print(f"    {source['text'][:150]}...")
            if source['metadata']:
                print(f"    Metadata: {source['metadata']}")


if __name__ == "__main__":
    example_rag_system()

    print("\n" + "=" * 70)
    print("✓ RAG system demonstration complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Use real embeddings (sentence-transformers)")
    print("  2. Integrate with vector DB (FAISS, Chroma)")
    print("  3. Connect to real LLM (OpenAI, Anthropic)")
    print("  4. Scale to thousands of papers")

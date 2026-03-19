import { RLMIteration, RLMLogFile, LogMetadata, RLMConfigMetadata, extractFinalAnswer } from './types';

// Extract the context variable from code block locals
export function extractContextVariable(iterations: RLMIteration[]): string | null {
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string') {
          return ctx;
        }
      }
    }
  }
  return null;
}

// Default config when metadata is not present (backwards compatibility)
function getDefaultConfig(): RLMConfigMetadata {
  return {
    root_model: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
  };
}

export interface ParsedJSONL {
  iterations: RLMIteration[];
  config: RLMConfigMetadata;
}

function hasConfig(config: RLMConfigMetadata): boolean {
  return (
    config.root_model !== null ||
    config.max_depth !== null ||
    config.max_iterations !== null ||
    config.backend !== null ||
    config.backend_kwargs !== null ||
    config.environment_type !== null ||
    config.environment_kwargs !== null ||
    config.other_backends !== null
  );
}

function configFromRunMetadata(runMetadata: Record<string, unknown>): RLMConfigMetadata {
  return {
    root_model: (runMetadata.root_model as string | null) ?? null,
    max_depth: (runMetadata.max_depth as number | null) ?? null,
    max_iterations: (runMetadata.max_iterations as number | null) ?? null,
    backend: (runMetadata.backend as string | null) ?? null,
    backend_kwargs: (runMetadata.backend_kwargs as Record<string, unknown> | null) ?? null,
    environment_type: (runMetadata.environment_type as string | null) ?? null,
    environment_kwargs: (runMetadata.environment_kwargs as Record<string, unknown> | null) ?? null,
    other_backends: (runMetadata.other_backends as string[] | null) ?? null,
  };
}

export function parseJSONL(content: string): ParsedJSONL {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const iterations: RLMIteration[] = [];
  let config: RLMConfigMetadata = getDefaultConfig();
  
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      
      // Raw logger metadata entries
      if (parsed.type === 'metadata') {
        config = {
          root_model: parsed.root_model ?? null,
          max_depth: parsed.max_depth ?? null,
          max_iterations: parsed.max_iterations ?? null,
          backend: parsed.backend ?? null,
          backend_kwargs: parsed.backend_kwargs ?? null,
          environment_type: parsed.environment_type ?? null,
          environment_kwargs: parsed.environment_kwargs ?? null,
          other_backends: parsed.other_backends ?? null,
        };
      // Generated SFT record wrapper format:
      // { ..., trace: { iterations: [...], run_metadata: {...} } }
      } else if (parsed.trace && Array.isArray(parsed.trace.iterations)) {
        iterations.push(...(parsed.trace.iterations as RLMIteration[]));

        if (!hasConfig(config) && parsed.trace.run_metadata && typeof parsed.trace.run_metadata === 'object') {
          config = configFromRunMetadata(parsed.trace.run_metadata as Record<string, unknown>);
        }
      // Direct trajectory wrapper format: { iterations: [...], run_metadata: {...} }
      } else if (Array.isArray(parsed.iterations)) {
        iterations.push(...(parsed.iterations as RLMIteration[]));

        if (!hasConfig(config) && parsed.run_metadata && typeof parsed.run_metadata === 'object') {
          config = configFromRunMetadata(parsed.run_metadata as Record<string, unknown>);
        }
      } else {
        // Raw logger iteration entry
        iterations.push(parsed as RLMIteration);
      }
    } catch (e) {
      console.error('Failed to parse line:', line, e);
    }
  }
  
  return { iterations, config };
}

export function extractContextQuestion(iterations: RLMIteration[]): string {
  if (iterations.length === 0) return 'No context found';
  
  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt;
  
  // Look for user message that contains the actual question
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      // Try to extract quoted query
      const queryMatch = msg.content.match(/original query: "([^"]+)"/);
      if (queryMatch) {
        return queryMatch[1];
      }
      
      // Check if it contains the actual query pattern
      if (msg.content.includes('answer the prompt')) {
        continue;
      }
      
      // Take first substantial user message
      if (msg.content.length > 50 && msg.content.length < 500) {
        return msg.content.slice(0, 200) + (msg.content.length > 200 ? '...' : '');
      }
    }
  }
  
  // Fallback: look in system prompt for context info
  const systemMsg = prompt.find(m => m.role === 'system');
  if (systemMsg?.content) {
    const contextMatch = systemMsg.content.match(/context variable.*?:(.*?)(?:\n|$)/i);
    if (contextMatch) {
      return contextMatch[1].trim().slice(0, 200);
    }
  }
  
  // Check code block output for actual context
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string' && ctx.length < 500) {
          return ctx;
        }
      }
    }
  }
  
  return 'Context available in REPL environment';
}

export function computeMetadata(iterations: RLMIteration[]): LogMetadata {
  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;
  
  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;
    
    // Use iteration_time if available, otherwise sum code block times
    if (iter.iteration_time != null) {
      totalExecutionTime += iter.iteration_time;
    } else {
      for (const block of iter.code_blocks) {
        if (block.result) {
          totalExecutionTime += block.result.execution_time || 0;
        }
      }
    }
    
    for (const block of iter.code_blocks) {
      if (block.result) {
        if (block.result.stderr) {
          hasErrors = true;
        }
        if (block.result.rlm_calls) {
          totalSubLMCalls += block.result.rlm_calls.length;
        }
      }
    }
    
    if (iter.final_answer) {
      finalAnswer = extractFinalAnswer(iter.final_answer);
    }
  }
  
  return {
    totalIterations: iterations.length,
    totalCodeBlocks,
    totalSubLMCalls,
    contextQuestion: extractContextQuestion(iterations),
    finalAnswer,
    totalExecutionTime,
    hasErrors,
  };
}

export function parseLogFile(fileName: string, content: string): RLMLogFile {
  const { iterations, config } = parseJSONL(content);
  const metadata = computeMetadata(iterations);
  
  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
  };
}

